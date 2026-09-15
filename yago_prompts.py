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
    "1. Ground the query in ONLY the resolved entities from the context; "
    "never invent other entity URIs.\n"
    "2. Declare EVERY prefix at the top of the query yourself - the Prefixes "
    "block in the context is reference material and is NOT applied "
    "automatically. Start with PREFIX yago: <http://yago-knowledge.org/resource/> "
    "and declare every namespace you use (schema: is http://schema.org/, "
    "never https).\n"
    "3. SELECT exactly the variables that constitute the answer, nothing more:\n"
    "   - Who/what questions: bind the value the question asks for. Prefer a "
    "string value (rdfs:label, schema:name) over an opaque URI; if the direct "
    "value is a URI and the question asks for a name, fetch its rdfs:label in "
    "an OPTIONAL block and select the label.\n"
    "   - How-many questions: SELECT (COUNT(...) AS ?count).\n"
    "   - When/where questions: bind the date or the place variable directly.\n"
    "   - Add LIMIT 20 unless the query aggregates.\n"
    "4. Keep patterns simple: one triple per fact, joined through the context "
    "entities. Prefer properties from the property list in the context when "
    "one fits; use rdfs:label and rdf:type freely for names and classes.\n"
    "5. Output ONLY the SPARQL query - no explanation, no markdown fences."
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
