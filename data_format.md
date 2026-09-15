# Training Data Format — Text2SPARQL (Yago)

One JSONL line per example. Chat format (Qwen3 native). ~10k examples.

## Example line

```json
{"messages": [
  {"role": "system", "content": "You are an expert SPARQL generator for the Yago knowledge graph. Convert the user's question into a valid SPARQL query. Use ONLY the entities listed in the context. Output ONLY the SPARQL query — no explanation, no commentary."},
  {"role": "user", "content": "Question: Who directed the film Inception?\n\nEntities:\n- Inception -> yago:Inception_(film)\n- directed -> <http://yago-knowledge.org/resource/wasDirectedBy>\n\nPrefixes:\nPREFIX yago: <http://yago-knowledge.org/resource/>\nPREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"},
  {"role": "assistant", "content": "PREFIX yago: <http://yago-knowledge.org/resource/>\nPREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nSELECT ?director WHERE { yago:Inception_(film) <http://yago-knowledge.org/resource/wasDirectedBy> ?director . }"}
]}
```

## Hard rules (these decide success or failure)

1. **Entity pre-resolution is mandatory.** Never make the model guess IDs.
   Run each question through a label->ID lookup (you already have roqet/Fuseki)
   and inject the resolved IDs into the prompt. The papers are unanimous:
   small models nail query *structure* but fail *entity linking*.
2. **Include prefixes in EVERY example.** Models copy prefixes reliably;
   don't make them memorize them.
3. **One query per example. No explanation.** Keep the output token space
   pure SPARQL.
4. **Balance query shapes:** SELECT, COUNT, FILTER, OPTIONAL, ORDER BY,
   LIMIT, DISTINCT, UNION, nested subqueries. If you only train on simple
   SELECTs, that's all you'll ever get.
5. **Mix simple and complex questions.** The known failure mode is complex
   multi-hop questions — include a solid chunk of them.
6. **No thinking mode.** Assistant content must be plain SPARQL. If your
   data pipeline ever wraps answers in `<|thinking_start|>`...`<|thinking_end|>`,
   the model learns to emit those tokens. (Trainer forces `enable_thinking: False`,
   and `validate.py` strips any thinking blocks that slip through before
   checking parseability — belt and suspenders.)

## Generating synthetic data (off-policy is fine for SFT)

- Use a strong LLM (GPT-4-class or a big open model) to generate
  question/query pairs against your Yago schema, then **validate every query**
  by executing it against Fuseki and keeping only ones that parse AND return
  the expected result.
- Paraphrase each question 2-3x for robustness (the AKSW paper did exactly this).
- The AKSW paper's data recipe: generate -> execute -> filter -> paraphrase.
  Their 69-131 example datasets worked; 10k is plenty generous.

## Existing reference checkpoints (DBpedia, not Yago)

- `aksw/text2sparql-S` and `aksw/text2sparql-M` (Qwen2.5-7B, Apache-2.0,
  trained with Unsloth+TRL). Use them as format/style references, not weights
  to build on — different ontology.
- Skip all T5-based checkpoints. The literature shows T5 family cannot
  produce valid SPARQL.

## Validation loop (do this every N steps)

Generate on a held-out set -> `rdflib.plugins.sparql.prepareQuery()` for
parseability -> execute against Fuseki -> compare result sets with gold.
Parseability alone is the 80% win; result-set match is the 20% that makes
it actually useful.