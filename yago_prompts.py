"""Shared prompt/namespace definitions for generate_data.py and validate.py.

Single source of truth so the fine-tuning prompt and the eval prompt can
never drift apart. Keep this module stdlib-only (no openai/rdflib imports).
"""

import re

YAGO_NS = "http://yago-knowledge.org/resource/"

PREFIXES = (
    "PREFIX yago: <http://yago-knowledge.org/resource/>\n"
    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"
)

# Known namespaces. Used by generate_data.py's ensure_prefixes() to
# auto-repair queries that use a prefix without declaring it.
KNOWN_NS = {
    "yago": YAGO_NS,
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "schema": "http://schema.org/",
    "wd": "http://www.wikidata.org/entity/",
    "wdt": "http://www.wikidata.org/prop/direct/",
}

SYSTEM_PROMPT = (
    "You are an expert SPARQL generator for the Yago knowledge graph "
    "(Yago4, schema.org vocabulary, entities under yago:).\n"
    "Rules:\n"
    "1. Bind every entity by NAME with an exact English label seek, copying "
    "the name EXACTLY as spelled in the question: "
    "?person rdfs:label \"Mario Scarabelli\"@en . NEVER paste entity URIs: "
    "URIs like yago:Mario_Scarabelli_Q3849089 carry opaque ID suffixes that "
    "cannot be derived from a question and must never be guessed.\n"
    "2. Add an rdf:type filter ONLY when the question itself names a role or "
    "category (athlete, actor, footballer, film, band...): "
    "?person rdf:type yago:Voice_actor . Graph class names are CamelCase "
    "with underscores between words (Voice_actor, Association_football_player, "
    "Politician, Film). If the question names no role or category, do NOT "
    "add any rdf:type filter.\n"
    "3. Declare EVERY prefix at the top of the query yourself. Start with "
    "PREFIX yago: <http://yago-knowledge.org/resource/> and declare every "
    "namespace you use (schema: is http://schema.org/, never https).\n"
    "4. SELECT exactly the variables that constitute the answer, nothing more:\n"
    "   - Who/what questions: bind the value the question asks for. Prefer a "
    "string value (rdfs:label, schema:name) over an opaque URI; if the direct "
    "value is a URI and the question asks for a name, fetch its rdfs:label in "
    "an OPTIONAL block and select the label. Always restrict labels to "
    "English with FILTER(LANG(?label) = \"en\") - the graph contains labels "
    "in dozens of languages.\n"
    "   - Nationality questions: schema:nationality points at a country node, "
    "but the natural answer is the demonym (\"Italian\", not \"Italy\"). Join "
    "the country node to yago:demonym and SELECT the demonym. Demonym values "
    "are plain multilingual strings with NO language tags - never apply a "
    "LANG filter to them.\n"
    "   - How-many questions: SELECT (COUNT(...) AS ?count).\n"
    "   - When/where questions: bind the date or the place variable directly.\n"
    "   - Add LIMIT 20 unless the query aggregates. Selecting yago:demonym "
    "counts as multilingual, too - use LIMIT 50 there.\n"
    "5. Keep patterns simple: one triple per fact, joined through the "
    "entities named in the question. Use rdfs:label and rdf:type freely "
    "for names and classes.\n"
    "6. Output ONLY the SPARQL query - no explanation, no markdown fences."
)

EXTRACT_SYSTEM = (
    "You extract entity mentions from natural language questions about the "
    "Yago knowledge graph. Entities are people, films, places, organizations, "
    "or other named things - NOT verbs, adjectives, or properties. "
    'Reply with a JSON array of strings only, e.g. ["Inception", "Christopher Nolan"].'
)

QUESTION_GEN_SYSTEM = (
    "You generate natural language questions that can be answered by querying "
    "the Yago knowledge graph. You are given an entity and a property that "
    "exist in the graph. Write a natural, varied question a person might ask "
    "that involves them. Vary the phrasing each time - different sentence "
    "structures, sometimes mention the entity, sometimes describe it. "
    "Reply with ONLY the question, no preamble."
)

ANSWER_SYSTEM = (
    "You answer questions about the Yago knowledge graph (a Wikipedia-derived "
    "snapshot). Answer concisely with the facts only - the entity's common "
    "English name, a number, a date, or a short list. Use the context "
    "entities to disambiguate which specific entity is meant. If you are not "
    "sure, say 'UNKNOWN'. Do not explain."
)

RESULTS_SYSTEM = (
    "You are a strict quality judge for SPARQL training data. You are given a "
    "natural-language question, the SPARQL query written for it, and the rows "
    "the query actually returned from the knowledge graph. Decide whether the "
    "query and its results answer THAT question.\n"
    "Verdicts:\n"
    '- "answer": the query reads the property/fact the question asks about '
    "AND the rows contain that fact. Extra variants (other languages, extra "
    "rows, list values) are fine.\n"
    '- "wrong": the query reads the wrong property (e.g. deathDate when the '
    "question asks for birthDate, actor when it asks for director), the rows "
    "are about a different entity than the question names, or the results "
    "plainly do not answer the question.\n"
    '- "unsure": you cannot tell from the question, query, and rows alone.\n'
    'Reply with ONLY JSON: {"verdict": "answer|wrong|unsure", "reason": "short"}'
)

# Properties seen in the Yago4 graph (schema.org vocabulary) + the two
# structural predicates every query needs. Grounds property choice instead
# of letting the model guess property names.
PROPERTY_HINTS = (
    "schema:director, schema:actor, schema:author, schema:creator, "
    "schema:composer, schema:producer, schema:spouse, schema:children, "
    "schema:birthPlace, schema:deathPlace, schema:nationality, "
    "schema:birthDate, schema:deathDate, schema:employer, schema:alumniOf, "
    "schema:knowsLanguage, schema:memberOf, schema:country, schema:capital, "
    "schema:location, schema:founder, schema:foundingDate, schema:genre, "
    "schema:inLanguage, schema:publisher, schema:publicationDate, "
    "schema:award, schema:language, schema:manufacturer, schema:ingredient, "
    "schema:parentTaxon, schema:follows, rdf:type, rdfs:label"
)


def strip_reasoning(text):
    """Remove thinking blocks and code fences so only the answer remains."""
    if not text:            # reasoning models can return None content
        return ""
    text = re.sub(r"<\|thinking_start\|>.*?<\|thinking_end\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:sparql)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
    return text.strip()
