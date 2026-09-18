#!/usr/bin/env python3
"""
Validate fine-tuned SPARQL output: parseability + exact-match against gold.

Target hardware: Intel Arc B65 (XPU, native PyTorch >=2.8 — IPEX is EOL).
Strips reasoning/thinking blocks BEFORE validating so Qwen3's thinking mode
can't pollute the SPARQL.

Kept in sync with train_sparql.py on purpose: same base model, prompt,
generation params, and MLflow tracking (sqlite:///mlflow.db). Eval numbers
from a mismatched setup lie about training quality.

Every row's expected vs actual output is dumped to an MLflow artifact
(eval/predictions_*.jsonl) so bad generations can be inspected run-over-run.

Usage:
    python validate.py --model output/qwen3-0.6b-sparql --eval data/eval.jsonl
    python validate.py --model output/qwen3-0.6b-sparql --eval data/eval.jsonl --limit 20
    python validate.py ... --no-exec              # parse/exact only
    python validate.py ... --fuseki http://localhost:3030/yago/sparql
    mlflow ui          # then open http://localhost:5000 to view runs

Prints: parse rate, exact-match rate, and (when Fuseki is reachable)
execution metrics — exec rate (query actually ran), execution accuracy
(predicted results == gold results), and empty-result rate. Per-row
exec_ok/exec_error/exec_match land in the predictions artifact.
"""

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import mlflow
import torch
from rdflib.plugins.sparql import prepareQuery
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from yago_prompts import SYSTEM_PROMPT, strip_reasoning

# --- Keep in sync with train_sparql.py --------------------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B"                # must match the training base
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"   # local, no server needed
MLFLOW_EXPERIMENT = "qwen3-0.6b-sparql"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.01    # near-greedy: SPARQL wants determinism
TOP_P = 0.9

FUSEKI_URL = "http://192.168.8.110:3030/yago/sparql"
EXEC_TIMEOUT = 30   # seconds per query against Fuseki


def generate(model, tokenizer, question):
    # Must match the training prompt EXACTLY (yago_prompts.py is the source
    # of truth) or eval numbers will lie about training quality. Training
    # user turns are the BARE question — no entities/properties/prefixes.
    # Also matches the post-train spot-check block in train_sparql.py.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        #chat_template_kwargs={"enable_thinking": False},  # Qwen3: no thinking mode
    ).to("xpu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=TOP_P,
        )
    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return strip_reasoning(decoded)


def is_parsable(query: str) -> bool:
    try:
        prepareQuery(query)
        return True
    except Exception:
        return False


# --- Execution checking (Fuseki) ---------------------------------------------
def fuseki_select(query: str, timeout: int = EXEC_TIMEOUT):
    """Run a SELECT against Fuseki, return raw bindings. Raises on any failure."""
    url = FUSEKI_URL + "?" + urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(url, headers={
        "Accept": "application/sparql-results+json",
        "User-Agent": "sparql-eval/0.1",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["results"]["bindings"]


def normalize_bindings(bindings) -> set:
    """Bindings -> comparable set of rows (each row: frozenset of (var, value)).

    Only the lexical form is compared, so "341000"^^xsd:integer == "341000".
    Variable names ARE part of the row — ?name vs ?label won't match, which is
    intentional: the system prompt's rule 4 makes the selected vars the contract.
    """
    return {
        frozenset((k, v.get("value", "")) for k, v in b.items())
        for b in bindings
    }


def fuseki_reachable() -> bool:
    """Probe once at startup so a down Fuseki degrades to parse/exact-only eval
    instead of crashing (or worse, counting connection errors as model errors)."""
    try:
        fuseki_select("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1", timeout=5)
        return True
    except Exception as e:
        print(f"WARNING: Fuseki unreachable at {FUSEKI_URL} ({e})")
        print("         execution metrics disabled for this run")
        return False


def main():
    global FUSEKI_URL, EXEC_TIMEOUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to saved PEFT adapter")
    ap.add_argument("--eval", required=True, help="eval JSONL path")
    ap.add_argument("--limit", type=int, default=0,
                    help="only eval the first N rows (0 = all)")
    ap.add_argument("--run-name", default=None, help="MLflow run name")
    ap.add_argument("--fuseki", default=FUSEKI_URL,
                    help=f"Fuseki SPARQL endpoint (default: {FUSEKI_URL})")
    ap.add_argument("--no-exec", action="store_true",
                    help="skip execution checking (parse/exact only)")
    ap.add_argument("--exec-timeout", type=int, default=EXEC_TIMEOUT,
                    help=f"seconds per Fuseki query (default: {EXEC_TIMEOUT})")
    args = ap.parse_args()

    FUSEKI_URL = args.fuseki
    EXEC_TIMEOUT = args.exec_timeout
    exec_enabled = (not args.no_exec) and fuseki_reachable()

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to("xpu")
    model = PeftModel.from_pretrained(base, args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    n = len(rows)
    if n == 0:
        raise SystemExit(f"no rows found in {args.eval}")

    parse_ok = exact_ok = exec_ok = match_ok = empty_ok = 0
    n_exec = n_matchable = 0
    records = []

    for row in rows:
        q = generate(model, tokenizer, row["question"])
        gold = row.get("gold_query") or ""
        parsable = is_parsable(q)
        exact = bool(gold) and q.strip() == gold.strip()
        parse_ok += parsable
        exact_ok += exact

        rec = {
            "question": row["question"],
            "expected": gold,
            "actual": q,
            "parsable": parsable,
            "exact": exact,
        }

        # Execution check: run the predicted query against Fuseki, and the gold
        # too, then compare result rows. This is the metric that catches
        # "parses fine, answers wrong" (or returns nothing at all).
        if exec_enabled:
            n_exec += 1
            try:
                pred_rows = normalize_bindings(fuseki_select(q))
                rec["exec_ok"] = True
                exec_ok += 1
                if not pred_rows:
                    rec["exec_empty"] = True
                    empty_ok += 1
            except Exception as e:
                rec["exec_ok"] = False
                rec["exec_error"] = str(e)[:200]
                pred_rows = None
            if gold:
                n_matchable += 1
                try:
                    gold_rows = normalize_bindings(fuseki_select(gold))
                except Exception as e:
                    gold_rows = None
                    rec["gold_exec_error"] = str(e)[:200]
                if pred_rows is not None and gold_rows is not None:
                    rec["exec_match"] = (pred_rows == gold_rows)
                    match_ok += rec["exec_match"]

        records.append(rec)
        line = f"[{len(records)}/{n}] parsable={parsable} exact={exact}"
        if exec_enabled:
            line += f" exec={rec.get('exec_ok')} match={rec.get('exec_match')}"
        print(line)

    parse_rate = parse_ok / n
    exact_rate = exact_ok / n

    # MLflow: same tracking setup as train_sparql.py, so eval runs show up
    # next to their training runs in the UI.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    run_name = args.run_name or f"validate-{os.path.basename(args.model.rstrip('/'))}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "adapter": args.model,
            "eval_path": args.eval,
            "n_rows": n,
            "base_model": MODEL_NAME,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "fuseki": FUSEKI_URL,
            "exec_enabled": exec_enabled,
        })
        mlflow.log_metrics({"parse_rate": parse_rate, "exact_rate": exact_rate})
        if exec_enabled and n_exec:
            metrics = {"exec_rate": exec_ok / n_exec}
            if n_matchable:
                metrics["execution_accuracy"] = match_ok / n_matchable
            metrics["exec_empty_rate"] = empty_ok / n_exec
            mlflow.log_metrics(metrics)

        # Actual-vs-expected dump: one JSON line per row, so failures are
        # browsable in the MLflow UI (run -> Artifacts -> eval/).
        pred_path = f"predictions_{run.info.run_id[:8]}.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        mlflow.log_artifact(pred_path, artifact_path="eval")
        #os.remove(pred_path)

    print(f"parse rate:   {parse_ok}/{n} ({100*parse_rate:.1f}%)")
    print(f"exact rate:   {exact_ok}/{n} ({100*exact_rate:.1f}%)")
    if exec_enabled and n_exec:
        print(f"exec rate:    {exec_ok}/{n_exec} ({100*exec_ok/n_exec:.1f}%)")
        if n_matchable:
            print(f"exec match:   {match_ok}/{n_matchable} "
                  f"({100*match_ok/n_matchable:.1f}%)")
        print(f"empty result: {empty_ok}/{n_exec} ({100*empty_ok/n_exec:.1f}%)")
    print(f"actual vs expected: MLflow artifact eval/predictions_*.jsonl "
          f"({MLFLOW_TRACKING_URI}; view with: mlflow ui)")


if __name__ == "__main__":
    main()
