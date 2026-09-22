#!/usr/bin/env python3
"""
Generate Text2SPARQL training examples for the Yago knowledge graph,
in the exact format specified in data_format.md.

Pipeline per question:
  1. Extract entity mentions with an LLM (OpenRouter)
  2. Resolve mentions -> Yago resource IDs via index-seek lookups
     (direct URI probe / owl:sameAs Q-ID / exact label or alternateName —
     never CONTAINS, which full-scans the label index on a 22GB graph)
     (entity pre-resolution is the #1 success factor for small models)
  3. Generate a SPARQL query with the LLM (resolved entities in prompt)
  4. Validate: rdflib parse + execute against Fuseki (must return results)
  5. Expected-answer check: LLM predicts the answer from the question alone,
     and the prediction must appear in the query's actual result set
     (semantic correctness gate — catches wrong-property/wrong-entity queries)
  6. Results-relevance judge: LLM sees question + query + ACTUAL result rows
     and rejects when they don't answer the question (backstop for the
     answer gate's UNKNOWN-accept and lenient substring matches)
  7. Up to MAX_TRIES attempts, feeding the rejection reason back to the LLM
  8. Write valid examples to train.jsonl (messages format) and eval.jsonl
     (flat format that validate.py expects)

Requirements:
    pip install openai SPARQLWrapper rdflib
    export OPENROUTER_API_KEY=sk-or-...

Usage:
    python generate_data.py --questions questions.txt
    python generate_data.py --generate-questions 100
    python generate_data.py --questions questions.txt --generate-questions 200
    # Questions are generated INLINE (one per iteration). (subject, predicate)
    # pairs come from a LOCAL POOL harvested from Fuseki once and cached to
    # data/entity_pool.json — sampling is O(1) per question, never a graph scan.
    # Add --no-answer-check to skip the answer gate and --no-results-check
    # to skip the relevance judge (faster, weaker).
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.parse

from yago_prompts import (YAGO_NS, PREFIXES, KNOWN_NS, SYSTEM_PROMPT,
                          EXTRACT_SYSTEM, QUESTION_GEN_SYSTEM, ANSWER_SYSTEM,
                          RESULTS_SYSTEM, PROPERTY_HINTS, strip_reasoning)

from openai import OpenAI
from rdflib.plugins.sparql import prepareQuery
from SPARQLWrapper import SPARQLWrapper, JSON

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_FUSEKI = "http://192.168.8.110:3030/yago/sparql"
MAX_TRIES = 3                                 # total attempts per question (1 initial + 2 retries)
EVAL_RATIO = 0.1                              # fraction of examples routed to eval.jsonl
SLEEP_BETWEEN_CALLS = 1.0                     # polite to OpenRouter rate limits
# RNG seeding is done in main(): pass --seed N for reproducible sampling,
# otherwise every run draws a fresh random seed (printed to the log).
QUERY_TIMEOUT = 120                           # seconds; aborts runaway Fuseki queries instead of hanging
REASONING_EFFORT = "low"                      # OpenRouter reasoning effort for thinking models

# Predicates that make boring questions — excluded from the entity pool.
STRUCTURAL_PREDICATES = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/2002/07/owl#sameAs",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2000/01/rdf-schema#seeAlso",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "http://schema.org/additionalType",
    "http://schema.org/mainEntityOfPage",
    "http://schema.org/url",
    "http://schema.org/description",
    "http://schema.org/alternateName",
}

MAX_SAMPLE_ATTEMPTS = 5   # tries to find a usable triple before giving up on one question

QUERY_LOG = None   # file handle for the JSONL query log (opened in main)


def log_event(event):
    """Append one JSON line to the query log (if enabled). Flushed per line
    so the file can be tailed while the run is live."""
    if QUERY_LOG is not None:
        event["ts"] = time.strftime("%H:%M:%S")
        QUERY_LOG.write(json.dumps(event) + "\n")
        QUERY_LOG.flush()

# Global runtime config (set from CLI args in main)
MODEL = DEFAULT_MODEL
FUSEKI_URL = DEFAULT_FUSEKI
POOL = []          # cached (subject_uri, predicate_uri) pairs
POOL_RING = []     # shuffled index ring so samples don't repeat until exhausted

# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("ERROR: set OPENROUTER_API_KEY (export OPENROUTER_API_KEY=sk-or-...)")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "https://github.com/",   # OpenRouter asks for these
        "X-Title": "yago-sparql-data-gen",
    },
)


def llm_chat(messages, temperature=0.1, max_tokens=2048):
    """One OpenRouter chat completion with basic 429/error retry.

    Reasoning models (GLM etc.) burn tokens on thinking before answering —
    keep max_tokens generous or the answer never gets emitted (content=None).
    Thinking effort comes from REASONING_EFFORT (default 'low'); empty
    content is treated as a retryable failure, retries get extra headroom."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens + attempt * 1024,
                extra_body={"reasoning": {"effort": REASONING_EFFORT}},
            )
            content = resp.choices[0].message.content
            if not content or not content.strip():
                finish = getattr(resp.choices[0], "finish_reason", "?")
                raise ValueError(f"empty content (finish_reason={finish})")
            return content
        except Exception as e:
            msg = str(e)
            wait = 10 if "429" in msg else 2 ** attempt
            print(f"  [llm retry {attempt + 1}/3 after {wait}s] {msg[:120]}")
            time.sleep(wait)
    raise RuntimeError("LLM call failed after 3 attempts")


# ---------------------------------------------------------------------------
# Fuseki helpers
# ---------------------------------------------------------------------------
def sparql_select(query):
    sparql = SPARQLWrapper(FUSEKI_URL)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(QUERY_TIMEOUT)   # never hang forever on a runaway query
    res = sparql.query().convert()
    return res["results"]["bindings"]


def check_fuseki():
    try:
        sparql_select("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
    except Exception as e:
        sys.exit(f"ERROR: cannot reach Fuseki at {FUSEKI_URL}: {e}")


QID_RE = re.compile(r"^Q\d+$", re.I)
QID_TAIL_RE = re.compile(r"\s*Q\d+\s*$", re.I)   # Yago4 tacks Q-IDs onto labels


def strip_qid_suffix(label):
    """'Vehkosjärvi Q54806425' -> 'Vehkosjärvi'; bare 'Q19677782' -> ''."""
    return QID_TAIL_RE.sub("", label.strip()).strip()


def title_case(name):
    """'vladimir k. zworykin' -> 'Vladimir K. Zworykin' (no Women'S mangling)."""
    return " ".join(w[:1].upper() + w[1:] for w in name.split())


def mention_to_uri(mention):
    """Wikipedia-title-style Yago URI candidate: spaces -> underscores."""
    tail = urllib.parse.quote(mention.strip().replace(" ", "_"),
                              safe="'()._~!$&*+,;=:@-")
    return YAGO_NS + tail


def yago_uri_encode(name):
    """Yago4's real URI escaping (observed in-graph): space -> '_', any other
    punctuation -> '_uXXXX_' hex (e.g. '.' -> '_u002E_', "'" -> '_u0027_')."""
    out = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        elif ch == " ":
            out.append("_")
        else:
            out.append("_u%04X_" % ord(ch))
    return YAGO_NS + "".join(out)


def resolve_qid(qid):
    """Bare Wikidata Q-ID -> Yago resource via owl:sameAs (exact object seek)."""
    wd = f"http://www.wikidata.org/entity/{qid.upper()}"
    q = (
        "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
        f"SELECT ?s WHERE {{ {{ ?s owl:sameAs <{wd}> }} "
        f"UNION {{ <{wd}> owl:sameAs ?s }} }} LIMIT 1"
    )
    rows = sparql_select(q)
    return rows[0]["s"]["value"] if rows else None


def uri_has_label(uri):
    """O(1) existence probe: does the resource carry an en/untagged label?"""
    q = (
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        f"SELECT ?label WHERE {{ <{uri}> rdfs:label ?label . "
        'FILTER(LANG(?label) = "en" || LANG(?label) = "") } LIMIT 1'
    )
    return bool(sparql_select(q))


def resolve_exact_literal(mention):
    """Exact label / alternateName match. Every branch is an index seek on
    (predicate, object) — NO CONTAINS, so no full-table scan."""
    esc = json.dumps(mention.strip())   # safely quoted SPARQL literal
    q = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
SELECT DISTINCT ?s ?label WHERE {{
  {{ ?s rdfs:label {esc}@en }}
  UNION {{ ?s rdfs:label {esc} }}
  UNION {{ ?s schema:alternateName {esc}@en }}
  UNION {{ ?s schema:alternateName {esc} }}
  FILTER(STRSTARTS(STR(?s), "{YAGO_NS}"))
}} ORDER BY STRLEN(STR(?label)) LIMIT 10
"""
    rows = sparql_select(q)
    return rows[0]["s"]["value"] if rows else None


def resolve_entity(mention):
    """Mention -> Yago resource URI (or None). Strategy (all index seeks,
    milliseconds each — the old CONTAINS version timed out on this dataset):
      1. bare Q-ID -> owl:sameAs lookup
      2. direct URI probe (as-written, then title-cased)
      3. exact rdfs:label / schema:alternateName match (as-written, then
         title-cased)"""
    m = mention.strip()
    if not m:
        return None
    candidates = list(dict.fromkeys([m, title_case(m)]))

    # 1. Bare Wikidata Q-ID
    if QID_RE.match(m):
        try:
            uri = resolve_qid(m)
            if uri:
                return uri
        except Exception as e:
            print(f"  [warn] Q-ID lookup failed for '{m}': {str(e)[:100]}")

    # 2. Direct URI probes (Yago4 hex-escaped form first, then plain)
    uri_candidates = []
    for base in candidates:
        uri_candidates += [yago_uri_encode(base), mention_to_uri(base)]
    for uri in dict.fromkeys(uri_candidates):
        try:
            if uri_has_label(uri):
                return uri
        except Exception as e:
            print(f"  [warn] URI probe failed for '{uri}': {str(e)[:100]}")

    # 3. Exact literal match
    for candidate in candidates:
        try:
            uri = resolve_exact_literal(candidate)
            if uri:
                return uri
        except Exception as e:
            print(f"  [warn] exact-label lookup failed for '{candidate}': {str(e)[:100]}")

    return None


def render_entity(uri):
    if uri.startswith(YAGO_NS):
        return "yago:" + uri[len(YAGO_NS):]
    return f"<{uri}>"


PREFIX_DECL_RE = re.compile(r"(?i)\bPREFIX\s+([A-Za-z_][\w-]*)\s*:\s*<([^>]*)>")


def ensure_prefixes(query):
    """Auto-repair the two prefix mistakes the LLM makes constantly:
      - https://schema.org/ instead of http://schema.org/ (parses fine,
        matches nothing -> silent 'no results')
      - using a prefix without declaring it (the model treats the context
        Prefixes block as auto-applied; Fuseki predeclares NOTHING)
    Returns (repaired_query, changed)."""
    original = query
    query = query.replace("https://schema.org/", "http://schema.org/")
    declared = {m.group(1).lower() for m in PREFIX_DECL_RE.finditer(query)}
    missing = [
        f"PREFIX {name}: <{uri}>"
        for name, uri in KNOWN_NS.items()
        if name not in declared and re.search(rf"(?<![\w-]){re.escape(name)}:", query)
    ]
    if missing:
        query = "\n".join(missing) + "\n" + query
    return query, query != original


def validate_query(query):
    """Returns (ok, feedback, rows). Parse check is STRICT (no initNs) so the
    local check matches what Fuseki will actually accept — with initNs the
    check used to pass undeclared prefixes that Fuseki then rejected."""
    if not query:
        return False, "Empty query.", None
    try:
        prepareQuery(query)
    except Exception as e:
        return False, f"SPARQL syntax error: {e}", None
    try:
        rows = sparql_select(query)
    except Exception as e:
        return False, f"Execution error: {e}", None
    if not rows:
        return False, "Query returned no results.", None
    return True, "", rows


# ---------------------------------------------------------------------------
# Expected-answer verification (semantic correctness gate)
# ---------------------------------------------------------------------------
def predict_answer(question, entities_block=None):
    """LLM predicts the answer from the question ALONE plus the resolved
    entities (for disambiguation) — never the query, so it can't just read
    its own SPARQL back."""
    user = f"Question: {question}"
    if entities_block:
        user += f"\n\nContext entities:\n{entities_block}"
    msgs = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]
    raw = strip_reasoning(llm_chat(msgs, temperature=0.0, max_tokens=1024))
    return raw.strip()


def normalize_answer(text):
    """Prediction -> set of comparable strings. Handles URI tails, lists,
    punctuation, case."""
    if not text or text.upper() == "UNKNOWN":
        return set()
    parts = re.split(r"[\n,;]", text)
    out = set()
    for p in parts:
        p = p.strip().strip(".-'\"")
        if not p:
            continue
        # URI tail: last path segment, underscores -> spaces
        tail = p.rstrip("/").rsplit("/", 1)[-1].rsplit("#", 1)[-1].replace("_", " ")
        out.add(tail.lower())
        out.add(p.lower())
    return out


def normalize_results(rows):
    """Fuseki bindings -> set of comparable strings (URI tails + literals)."""
    out = set()
    for r in rows:
        for v in r.values():
            val = v["value"]
            if v.get("type") == "uri":
                tail = val.rstrip("/").rsplit("/", 1)[-1].rsplit("#", 1)[-1].replace("_", " ")
                out.add(tail.lower())
            else:
                out.add(val.strip().lower())
    return out


def check_expected_answer(question, query, rows, entities_block=None):
    """Returns (ok, feedback). Prediction must appear in the query results.
    An UNKNOWN/empty prediction is 'unverifiable', not 'wrong' — the query
    already executed and returned rows, so we accept instead of rejecting."""
    prediction = predict_answer(question, entities_block)
    if not prediction or prediction.strip(".! ").upper() == "UNKNOWN":
        return True, ""

    # Yes/no questions: a non-empty result set IS the affirmative answer.
    # (Rows are guaranteed non-empty here — empty results are rejected
    # earlier by validate_query.) Without this, "Yes" vs ["Order of..."]
    # poisons the retry loop into trying to SELECT the literal "Yes".
    norm_pred = prediction.strip(".,;:! \t").lower()
    if norm_pred in ("yes", "y", "yeah", "yep", "true"):
        return True, ""
    if norm_pred in ("no", "n", "nope", "false"):
        return False, (
            "Expected answer was 'no' but the query returned rows — the "
            "question is yes/no phrased; check whether the pattern matches "
            "the question's intent"
        )

    # Aggregate queries: compare numerically
    if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", query, re.I):
        pred_nums = re.findall(r"-?\d+(?:\.\d+)?", prediction)
        result_nums = re.findall(r"-?\d+(?:\.\d+)?", " ".join(v["value"] for r in rows for v in r.values()))
        if pred_nums and result_nums and abs(float(pred_nums[0]) - float(result_nums[0])) < 1e-6:
            return True, ""
        return False, (
            f"Expected answer mismatch: predicted '{prediction}', "
            f"query returned {result_nums[:5]}"
        )

    pred_set = normalize_answer(prediction)
    result_set = normalize_results(rows)
    if pred_set & result_set:
        return True, ""

    # Lenient pass: prediction is a superset mention of a result (e.g. "the film Inception")
    for p in pred_set:
        for r in result_set:
            if p in r or r in p:
                return True, ""

    sample = sorted(result_set)[:5]
    return False, (
        f"Expected answer mismatch: predicted '{prediction}', "
        f"query returned {sample}"
    )


def check_results_relevant(question, query, rows):
    """Results-relevance judge — backstop for check_expected_answer's blind
    spots. The expected-answer gate trusts the predictor: an UNKNOWN
    prediction is ACCEPTED unverified, and lenient substring matches can pass
    by accident, so a gold query whose rows answer a different question can
    survive. This judge sees the question, the query, and the ACTUAL rows and
    rejects when they don't line up (wrong property, wrong entity).
    'unsure' accepts: same accept-not-reject philosophy as UNKNOWN above, and
    it keeps genuinely ambiguous questions from burning the retry budget.
    Only called with non-empty rows (validate_query rejects empties first)."""
    user = (
        f"Question: {question}\n\nQuery:\n{query}\n\n"
        f"Result rows:\n{rows_preview(rows, limit=8)}"
    )
    msgs = [
        {"role": "system", "content": RESULTS_SYSTEM},
        {"role": "user", "content": user},
    ]
    raw = strip_reasoning(llm_chat(msgs, temperature=0.0, max_tokens=512))
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return True, ""   # unparseable judge output -> don't reject on a judge malfunction
    try:
        verdict = json.loads(m.group(0))
    except json.JSONDecodeError:
        return True, ""
    v = str(verdict.get("verdict", "unsure")).strip().lower()
    reason = str(verdict.get("reason", ""))[:300]
    log_event({"type": "results_judge", "question": question,
               "verdict": v, "reason": reason})
    if v == "wrong":
        return False, (
            f"Results do not answer the question: {reason}. Check that the "
            "property and pattern match what the question actually asks for."
        )
    return True, ""       # "answer" and "unsure" both pass


# ---------------------------------------------------------------------------
# Per-question pipeline
# ---------------------------------------------------------------------------
def extract_mentions(question):
    msgs = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": f"Question: {question}"},
    ]
    raw = strip_reasoning(llm_chat(msgs, temperature=0.0, max_tokens=1024))
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        mentions = json.loads(m.group(0))
        return [x for x in mentions if isinstance(x, str) and x.strip()]
    except json.JSONDecodeError:
        return re.findall(r'"([^"]+)"', m.group(0))


def rows_preview(rows, limit=5):
    """Compact rendering of result rows for the retry prompt."""
    lines = []
    for row in rows[:limit]:
        lines.append("; ".join(f"{var}={binding['value'][:80]}"
                               for var, binding in row.items()))
    return "\n".join(lines) if lines else "(no rows)"


def generate_sparql(question, entities_block, error=None, attempt=0,
                    prev_query=None, prev_rows=None):
    user = (
        f"Question: {question}\n\nEntities:\n{entities_block}\n\n"
        f"Properties:\n{PROPERTY_HINTS}\n\nPrefixes:\n{PREFIXES}"
    )
    if error:
        user += f"\n\nYour previous attempt was rejected:\n{error}"
    if prev_query:
        user += f"\n\nYour previous query was:\n{prev_query}"
    if prev_rows:
        user += (
            f"\n\nThat query returned:\n{rows_preview(prev_rows)}\n\n"
            "If those results actually answer the question, adapt the query "
            "minimally (e.g. fix the SELECT clause). If not, fix the property "
            "or pattern."
        )
    user += "\n\nOutput ONLY the SPARQL query."
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    raw = llm_chat(msgs, temperature=0.1, max_tokens=1024)
    query = strip_reasoning(raw)
    query, repaired = ensure_prefixes(query)
    log_event({
        "type": "sparql_gen",
        "question": question,
        "attempt": attempt,
        "raw_output": raw,
        "query": query,
        "prefix_repair": repaired,
    })
    return query


# ---------------------------------------------------------------------------
# Question generation (grounded in the actual KG, sampled from a local pool)
# ---------------------------------------------------------------------------
def discover_predicates(limit):
    """DISTINCT short-circuits once `limit` distinct predicates are found, so
    this stays cheap even on a billion-triple graph. IMPORTANT: keep `limit`
    modest — a limit larger than the real vocabulary count degrades into a
    full-graph scan (minutes+)."""
    try:
        rows = sparql_select(
            f"SELECT DISTINCT ?p WHERE {{ ?s ?p ?o }} LIMIT {int(limit)}")
    except Exception as e:
        sys.exit(f"ERROR: predicate discovery failed: {e}")
    return [r["p"]["value"] for r in rows if r["p"]["value"] not in STRUCTURAL_PREDICATES]


def harvest_pool(pool_size, n_predicates, chunk):
    """One-time Fuseki harvest of (subject, predicate) pairs.
    Per predicate: subjects at offset 0 (indexed, instant) plus one random
    bounded offset (~<=1s index walk) for variety. Cost is bounded by the
    per-predicate index slice, never a full-graph scan."""
    preds = discover_predicates(n_predicates)
    if not preds:
        sys.exit("ERROR: no predicates discovered from Fuseki — is the dataset loaded?")
    random.shuffle(preds)
    print(f"  harvesting from {len(preds)} predicates...")
    pool, seen = [], set()
    for p in preds:
        if len(pool) >= pool_size:
            break
        for off in (0, random.randint(0, 20000) * 10):   # up to 200k into the slice
            try:
                rows = sparql_select(
                    f"SELECT ?s WHERE {{ ?s <{p}> ?o }} LIMIT {chunk} OFFSET {off}")
            except Exception as e:
                print(f"  [warn] harvest failed for {p} @+{off}: {str(e)[:100]}")
                continue
            for r in rows:
                s = r["s"]["value"]
                if not s.startswith(YAGO_NS):
                    continue
                key = (s, p)
                if key not in seen:
                    seen.add(key)
                    pool.append(key)
    return pool


def ensure_pool(args):
    """Build the pool once, cache to disk, reuse on every later run.
    After this, question sampling costs ZERO Fuseki scans."""
    global POOL, POOL_RING
    if not args.refresh_pool and os.path.exists(args.pool_file):
        try:
            with open(args.pool_file) as f:
                cache = json.load(f)
            if cache.get("fuseki") == args.fuseki and cache.get("pool"):
                POOL = [tuple(x) for x in cache["pool"]]
                print(f"Loaded entity pool: {len(POOL)} pairs from {args.pool_file} "
                      f"(built {cache.get('built_at', '?')})")
        except Exception as e:
            print(f"[warn] pool cache unreadable ({e}); rebuilding")
    if not POOL:
        print(f"Building entity pool (one-time harvest, cached to {args.pool_file})...")
        t0 = time.time()
        POOL = harvest_pool(args.pool_size, args.discover_predicates, args.pool_chunk)
        if not POOL:
            sys.exit("ERROR: harvested an empty entity pool — check Fuseki contents")
        with open(args.pool_file, "w") as f:
            json.dump({"fuseki": args.fuseki,
                       "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "pool": [list(x) for x in POOL]}, f)
        print(f"  {len(POOL)} (subject, predicate) pairs in {time.time() - t0:.1f}s")
    POOL_RING = list(range(len(POOL)))
    random.shuffle(POOL_RING)


def sample_triple():
    """Pop a random (subject, predicate) from the local pool. O(1), no Fuseki.
    The ring reshuffles only after the whole pool has been used once."""
    if not POOL:
        return None, None
    if not POOL_RING:
        POOL_RING[:] = list(range(len(POOL)))
        random.shuffle(POOL_RING)
    return POOL[POOL_RING.pop()]


def fetch_label(uri):
    q = (
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        f"SELECT ?label WHERE {{ <{uri}> rdfs:label ?label . "
        'FILTER(LANG(?label) = "en" || LANG(?label) = "") } LIMIT 1'
    )
    try:
        rows = sparql_select(q)
        if rows:
            return rows[0]["label"]["value"]
    except Exception:
        pass
    # Fallback: last path segment, underscores -> spaces, camelCase split
    tail = uri.rstrip("/").rsplit("/", 1)[-1].replace("_", " ")
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", tail)


NOISY_TYPES = {
    "http://www.w3.org/2002/07/owl#Thing",
    "http://schema.org/Thing",
}


def fetch_types(uri, limit=3):
    """rdf:type probe for the entity block. Returns up to `limit` concrete
    class URIs (rendered by the caller via render_entity, so the LLM sees
    stable `yago:Voice_actor`-style names it may copy into rdf:type filters
    verbatim). Root classes like owl:Thing are filtered out."""
    q = (
        "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
        f"SELECT ?t WHERE {{ <{uri}> rdf:type ?t }} LIMIT {limit + len(NOISY_TYPES) + 5}"
    )
    try:
        rows = sparql_select(q)
    except Exception as e:
        print(f"  [warn] type probe failed for '{uri}': {str(e)[:100]}")
        return []
    types = []
    for r in rows:
        t = r["t"]["value"]
        if t not in NOISY_TYPES and t not in types:
            types.append(t)
        if len(types) >= limit:
            break
    return types


def generate_question(entity_label, prop_label):
    """LLM writes a natural question involving the sampled entity+property."""
    msgs = [
        {"role": "system", "content": QUESTION_GEN_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Entity: {entity_label}\nProperty: {prop_label}\n\n"
                "Write a question about this."
            ),
        },
    ]
    q = strip_reasoning(llm_chat(msgs, temperature=0.8, max_tokens=1024))
    return q.strip().rstrip("?") + "?" if q else ""


def iter_questions(args):
    """Yield (question, source) pairs — the unified question source.

    File questions first (if --questions given), then inline-generated ones
    (if --generate-questions N): each is sampled live from Fuseki right when
    the loop needs it, so a bad/unusable sample is replaced immediately
    instead of poisoning a pre-generated batch.
    """
    if args.questions:
        for q in load_questions(args.questions):
            yield q, "file"
    if args.generate_questions:
        produced = 0
        attempts = 0
        max_attempts = args.generate_questions * MAX_SAMPLE_ATTEMPTS
        while produced < args.generate_questions and attempts < max_attempts:
            attempts += 1
            s, p = sample_triple()
            if not s or not p:
                continue
            s_label = strip_qid_suffix(fetch_label(s))
            p_label = strip_qid_suffix(fetch_label(p))
            # Q-ID questions are gone entirely: gold queries anchor on exact
            # rdfs:label names, and a bare "Q123" entity has no usable name.
            # Skip any sample whose label is empty after Q-ID stripping.
            if not s_label or not p_label:
                continue
            try:
                q = generate_question(s_label, p_label)
            except Exception as e:
                print(f"  [warn] question generation failed, skipping sample: {str(e)[:100]}")
                continue
            if q:
                produced += 1
                yield q, "generated"
            time.sleep(SLEEP_BETWEEN_CALLS)


def process_question(question, do_answer_check=True, do_results_check=True):
    """Returns (example_or_None, status_string)."""
    mentions = extract_mentions(question)
    if not mentions:
        return None, "no entity mentions extracted"

    # Resolve mentions -> URIs; dedupe by URI (keep first mention).
    resolved = {}
    for m in mentions:
        try:
            uri = resolve_entity(m)
        except Exception as e:
            print(f"  [warn] resolution of '{m}' failed: {str(e)[:100]}")
            continue
        if uri and uri not in resolved:
            resolved[uri] = m
        elif not uri:
            print(f"  [warn] could not resolve '{m}'")

    if not resolved:
        return None, "no mentions resolved to Yago entities"

    # Entity block: canonical NAME + types only — NO opaque URIs. This block
    # is GENERATOR SCAFFOLDING: the OpenRouter LLM sees it when writing the
    # gold query, but it is NEVER written into the training example (the
    # user turn is the bare question). The Q-ID suffix is stripped from the
    # name: entities whose stored label carries a Q-ID fail the consistency
    # gate below — by design.
    block_lines = []
    for uri, mention in resolved.items():
        lab = fetch_label(uri)
        typs = fetch_types(uri)
        name = strip_qid_suffix(lab or "") or mention
        # Plain-question gate: the fine-tuned model sees ONLY the question,
        # so the gold query's label anchor must be derivable from the
        # question text. A canonical name the question doesn't contain
        # (paraphrased/mangled spelling) would teach the model to invent
        # name forms — reject instead.
        if name.lower() not in question.lower():
            return None, (f'canonical name "{name}" missing from question '
                          f"(mention was '{mention}')")
        types = typs if typ_ok else []
        entry = f'- "{name}"'
        if types:
            entry += " (types: " + ", ".join(render_entity(t) for t in types) + ")"
        block_lines.append(entry)
    entities_block = "\n".join(block_lines)

    last_error = None
    last_query = None
    last_rows = None
    for attempt in range(1, MAX_TRIES + 1):
        query = generate_sparql(question, entities_block, error=last_error,
                                attempt=attempt, prev_query=last_query,
                                prev_rows=last_rows)
        try:
            ok, feedback, rows = validate_query(query)
        except Exception as e:
            ok, feedback, rows = False, f"validation error: {e}", None
        log_event({
            "type": "validation",
            "question": question,
            "attempt": attempt,
            "ok": ok,
            "feedback": feedback[:300],
        })
        if ok and do_answer_check:
            # Semantic gate: does the query actually answer the question?
            ok, feedback = check_expected_answer(question, query, rows, entities_block)
        if ok and do_results_check:
            # Relevance gate: judge sees question + query + ACTUAL rows, so it
            # catches wrong-data gold queries the answer gate let through
            # (UNKNOWN predictions are accepted unverified there).
            ok, feedback = check_results_relevant(question, query, rows)
        if ok:
            # Plain-language user turn: ONLY the question. The Entities/
            # Properties/Prefixes blocks exist purely for the generator LLM;
            # the fine-tuned model must map a bare question to SPARQL.
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": query},
                ]
            }
            return example, f"ok (attempt {attempt})"
        last_error = feedback
        last_query = query
        last_rows = rows or None   # [] on empty results -> no preview to show
        print(f"  [attempt {attempt}/{MAX_TRIES}] rejected: {feedback[:120]}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    return None, f"failed after {MAX_TRIES} attempts: {last_error}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_questions(path):
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                questions.append(json.loads(line)["question"])
            else:
                questions.append(line)
    return questions


def main():
    global MODEL, FUSEKI_URL, MAX_TRIES, SLEEP_BETWEEN_CALLS, QUERY_TIMEOUT, REASONING_EFFORT, QUERY_LOG

    ap = argparse.ArgumentParser(description="Generate Text2SPARQL training data for Yago")
    ap.add_argument("--questions", default=None,
                    help="txt (one question per line) or jsonl ({\"question\": ...}); "
                         "omit when using --generate-questions")
    ap.add_argument("--generate-questions", type=int, default=0,
                    help="generate N questions INLINE from live Fuseki samples "
                         "(after any --questions file is exhausted)")
    ap.add_argument("--out-questions", default=None,
                    help="optional: append every processed question to this file")
    ap.add_argument("--out-train", default="data/train.jsonl")
    ap.add_argument("--out-eval", default="data/eval.jsonl")
    ap.add_argument("--out-failed", default="data/failed.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fuseki", default=DEFAULT_FUSEKI)
    ap.add_argument("--max-tries", type=int, default=MAX_TRIES)
    ap.add_argument("--eval-ratio", type=float, default=EVAL_RATIO)
    ap.add_argument("--limit", type=int, default=0, help="max questions to process (0 = all)")
    ap.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_CALLS)
    ap.add_argument("--no-answer-check", action="store_true",
                    help="skip the expected-answer verification (faster, weaker filter)")
    ap.add_argument("--no-results-check", action="store_true",
                    help="skip the results-relevance judge (faster, weaker filter)")
    ap.add_argument("--pool-file", default="data/entity_pool.json",
                    help="cache file for the harvested (subject, predicate) pool")
    ap.add_argument("--pool-size", type=int, default=4000,
                    help="target number of pool pairs to harvest")
    ap.add_argument("--discover-predicates", type=int, default=60,
                    help="DISTINCT-predicate short-circuit limit; larger risks a "
                         "full-graph scan during pool harvest")
    ap.add_argument("--pool-chunk", type=int, default=40,
                    help="subjects fetched per predicate per offset during harvest")
    ap.add_argument("--refresh-pool", action="store_true",
                    help="ignore/rebuild the cached entity pool")
    ap.add_argument("--query-timeout", type=int, default=QUERY_TIMEOUT,
                    help="Fuseki query timeout in seconds")
    ap.add_argument("--reasoning-effort", default=REASONING_EFFORT,
                    choices=["low", "medium", "high"],
                    help="reasoning effort for thinking models (GLM etc.)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for reproducible pool harvest + sampling "
                         "(default: fresh random seed each run, printed at startup)")
    ap.add_argument("--query-log", default="data/query_log.jsonl",
                    help="JSONL log of every generated query + validation "
                         "feedback (append mode; empty string disables)")
    args = ap.parse_args()

    MODEL = args.model
    FUSEKI_URL = args.fuseki
    MAX_TRIES = args.max_tries
    SLEEP_BETWEEN_CALLS = args.sleep
    QUERY_TIMEOUT = args.query_timeout
    REASONING_EFFORT = args.reasoning_effort

    if not args.questions and not args.generate_questions:
        sys.exit("ERROR: provide --questions FILE and/or --generate-questions N")

    # Seed BEFORE anything touches `random` (pool harvest, ring shuffle,
    # offsets, eval split all draw from it). No --seed -> fresh entropy each
    # run, but the drawn seed is logged so any run can be replayed.
    if args.seed is None:
        args.seed = random.SystemRandom().randrange(2**32)
    random.seed(args.seed)
    print(f"RNG seed: {args.seed}  (replay with --seed {args.seed})")

    check_fuseki()
    if args.generate_questions:
        ensure_pool(args)

    os.makedirs(os.path.dirname(args.out_train) or ".", exist_ok=True)
    f_train = open(args.out_train, "w")
    f_eval = open(args.out_eval, "w")
    f_failed = open(args.out_failed, "w")
    f_questions = open(args.out_questions, "w") if args.out_questions else None
    if args.query_log:
        QUERY_LOG = open(args.query_log, "a")
        print(f"Logging queries to {args.query_log}")

    print(f"Processing with {MODEL} (max {MAX_TRIES} tries each)")
    print(f"~{1 + 3 * MAX_TRIES} LLM calls per question "
          f"(1 extract + up to {MAX_TRIES} generate + up to {MAX_TRIES} "
          f"answer-predict + up to {MAX_TRIES} results-judge)\n")

    stats = {"ok": 0, "failed": 0, "skipped": 0, "generated": 0}
    processed = 0
    try:
        for question, source in iter_questions(args):
            if args.limit and processed >= args.limit:
                break
            processed += 1
            if source == "generated":
                stats["generated"] += 1
            print(f"[{processed}] ({source}) {question[:80]}")
            if f_questions:
                f_questions.write(question + "\n")
            example, status = process_question(
                question,
                do_answer_check=not args.no_answer_check,
                do_results_check=not args.no_results_check,
            )
            if example is None:
                stats["skipped" if "no " in status else "failed"] += 1
                f_failed.write(json.dumps({"question": question, "reason": status}) + "\n")
                print(f"  -> {status}\n")
                continue
            line = json.dumps(example)
            if random.random() < args.eval_ratio:
                f_eval.write(json.dumps({
                    "question": question,
                    "gold_query": example["messages"][2]["content"],
                }) + "\n")
            else:
                f_train.write(line + "\n")
            stats["ok"] += 1
            print(f"  -> {status}\n")
            time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\nInterrupted — progress saved so far.")
    finally:
        f_train.close()
        f_eval.close()
        f_failed.close()
        if f_questions:
            f_questions.close()
        if QUERY_LOG is not None:
            QUERY_LOG.close()

    print("=" * 50)
    print(f"ok:        {stats['ok']}")
    print(f"failed:    {stats['failed']}")
    print(f"skipped:   {stats['skipped']}")
    print(f"generated: {stats['generated']}")
    print(f"train -> {args.out_train}\neval  -> {args.out_eval}\nfailed -> {args.out_failed}")


if __name__ == "__main__":
    main()