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
    "You generate natural language questions for the Yago knowledge graph. "
    "You are given one entity and ONE property that is stored for it in the "
    "graph. Write a natural, varied question a person might ask whose answer "
    "is exactly the value of THAT property for that entity. The question "
    "must be answerable by a single simple lookup of that property - never "
    "multi-hop reasoning, never about the database itself.\n"
    "Rules:\n"
    "1. The expected answer is what the property stores: an entity name, a "
    "date, a number, a measurement, or a code (ISBN, IATA/ICAO code, postal "
    "code). Phrase the question so that value is the natural answer.\n"
    "2. For numeric and code properties ask 'What is the ... of X' or 'How "
    "many/much/long ...' - the answer is a number or code, not a name "
    "(population, area, elevation, length, mass, luminosity, parallax, "
    "distance, HDI, unemployment rate, duration, ISBN, IATA code, postal "
    "code, jersey number, student count).\n"
    "3. Word the relation precisely so it cannot be confused with a similar "
    "property: birthPlace ('Where was X born?') vs deathPlace ('Where did X "
    "die?'); birthDate vs deathDate; author ('Who wrote X?') vs director "
    "('Who directed X?') vs editor; knowsLanguage ('What languages does X "
    "speak?') vs officialLanguage ('What language is spoken in X?'); "
    "neighbors ('Which countries border X?'); flowsInto ('Into which river "
    "does X flow?'); terminus ('Where does this line terminate?'); replaces "
    "('What did X replace?'); consumes ('What does X consume?'); candidateIn "
    "('In which election was X a candidate?'); conferredBy ('Who presents "
    "the X award?'); parentBody; sportNumber ('What number does X wear?'); "
    "notableWork; studentOf; doctoralAdvisor ('Who was X's doctoral "
    "advisor?').\n"
    "4. Do NOT ask anything the property cannot answer: no opinions, no "
    "'why' or 'how' reasoning, no superlatives or comparisons, no database "
    "questions (record counts, duplicates), no questions whose answer lives "
    "under a different property.\n"
    "5. Vary the phrasing each time - different sentence structures, "
    "sometimes name the entity, sometimes describe it ('the 1994 film about "
    "a heist'). Reply with ONLY the question, no preamble."
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

ANSWER_CHECK_SYSTEM = (
    "You verify answers for SPARQL training data over the Yago knowledge "
    "graph (a Wikipedia-derived snapshot). You are given a question, the "
    "SPARQL query written for it, and the rows the query returned from the "
    "graph. Decide:\n"
    '- "correct": general knowledge confirms the rows give the right answer '
    "to the question. Extra variants, other languages, or additional list "
    "entries are fine.\n"
    '- "wrong": the rows contradict general knowledge, are about a different '
    "entity than the question names, or the query reads a different property "
    "than what the question asks for (e.g. birth date when it asks for death "
    "date).\n"
    '- "plausible": the fact is too obscure to verify from general '
    "knowledge, BUT the query clearly reads the property the question asks "
    "about, anchors the entities it names, and the returned rows are a "
    "sensible answer. Use this for obscure-but-real questions where the "
    "query pattern is right.\n"
    '- "unanswerable": the question is not a general-knowledge question at '
    "all - it asks about the database itself (duplicates, record counts, "
    "what the dataset contains).\n"
    'Reply with ONLY JSON: {"verdict": "correct|wrong|plausible|unanswerable", '
    '"reason": "short"}'
)

# Properties seen in the Yago4 graph (schema.org vocabulary) + the two
# structural predicates every query needs. Grounds property choice instead
# of letting the model guess property names.
PROPERTY_HINTS = (
    # All predicate names verified against yago-4.5.0.2 yago-schema.ttl
    # (SHACL sh:path declarations). Phantom YAGO3-era names removed.
    "schema:director, schema:actor, schema:author, schema:musicBy, "
    "schema:productionCompany, schema:performer, schema:recordLabel, "
    "schema:lyricist, schema:editor, schema:illustrator, "
    "schema:spouse, schema:children, schema:birthPlace, schema:deathPlace, "
    "schema:nationality, schema:birthDate, schema:deathDate, "
    "schema:worksFor, schema:alumniOf, schema:knowsLanguage, "
    "schema:memberOf, schema:founder, schema:foundingDate, schema:location, "
    "schema:publisher, schema:dateCreated, schema:award, schema:inLanguage, "
    "schema:manufacturer, schema:organizer, schema:parentTaxon, "
    "schema:startDate, schema:endDate, schema:dissolutionDate, "
    "schema:duration, schema:numberOfSeasons, schema:numberOfEpisodes, "
    "schema:numberOfEmployees, schema:owns, "
    "yago:capital, yago:follows, yago:officialLanguage, yago:appearsIn, "
    "yago:playsIn, yago:notableWork, yago:ownedBy, yago:leader, "
    "yago:influencedBy, yago:studentOf, yago:doctoralAdvisor, "
    "yago:academicDegree, yago:participant, yago:neighbors, "
    "yago:flowsInto, yago:parentBody, yago:conferredBy, "
    "yago:beliefSystem, yago:demonym, yago:populationNumber, yago:area, "
    "yago:elevation, yago:length, yago:mass, yago:studentsCount, "
    "yago:unemploymentRate, yago:distanceFromEarth, yago:terminus, "
    "yago:replaces, yago:consumes, yago:candidateIn, "
    "yago:highestPoint, yago:lowestPoint, yago:sportNumber, yago:director, "
    "yago:parallax, yago:radialVelocity, yago:luminosity, "
    "yago:humanDevelopmentIndex, "
    "schema:address, schema:postalCode, schema:slogan, schema:sponsor, "
    "schema:locationCreated, schema:superEvent, schema:about, "
    "schema:affiliation, schema:material, schema:homeLocation, "
    "schema:contentLocation, schema:gender, schema:numberOfPages, "
    "schema:iataCode, schema:icaoCode, schema:isbn, rdf:type, rdfs:label"
)


def strip_reasoning(text):
    """Remove thinking blocks and code fences so only the answer remains."""
    if not text:            # reasoning models can return None content
        return ""
    text = re.sub(r"<\|thinking_start\|>.*?<\|thinking_end\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:sparql)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
    return text.strip()
