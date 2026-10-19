import asyncio, re, aiohttp
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import time
import mlflow

from yago_prompts import SYSTEM_PROMPT
from validate import strip_reasoning

FUSEKI = "http://192.168.8.110:3030/yago/sparql"
MODEL = "./output/qwen3.5-0.8b-sparql/checkpoint-294/"
OUTPUT_DIR = "output/qwen3.5-0.8b-sparql-rl"

# MLflow
#MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"   # local, no server needed
MLFLOW_TRACKING_URI = "http://192.168.8.110:5000/"
MLFLOW_EXPERIMENT = "qwen3.5-0.8b-sparql-rl"

# ---------------------------------------------------------------------------
# MLflow run: log config, train, log artifacts + a quick parse-rate check
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

async def execute_sparql(query):
    """None = failed. set() = ran (may be empty)."""
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(FUSEKI, params={"query": query, "timeout": 10000},
                             timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status != 200: return None
                data = await r.json()
        return {tuple(sorted(v["value"] for v in row.values()))
                for row in data["results"]["bindings"]}
    except Exception:
        return None

async def jena_reward(completions, gold_queries, **kwargs):
    async def score(completion, gold_query):
        if completion is None:
            return -0.5

        gold = await execute_sparql(gold_query["content"]) # Get actual expected result.

        start_time = time()
        pred = await execute_sparql(completion[0]["content"])
        duration_s = time() - start_time

        if pred is None:
            # Query failed to run.
            return -0.1 #return 0.0

        if not pred and not gold:
            # No results, intended.
            return 0.5

        if not pred:
            # No results, not intended.
            return 0.1

        if gold is None or not gold:
            # Nothing to compare to.
            return 0.0

        return (0.4 + 0.6 * len(pred & gold) / len(pred | gold)) * ((10. / (10. + duration_s)) ** 2)

    return await asyncio.gather(*(score(c, g) for c, g in zip(completions, gold_queries)))

def convert_ds(example):
    msgs = example["messages"]
    sysmsg = msgs[0]
    usrmsg = msgs[1]
    query = msgs[2]
    return {
        "prompt": [sysmsg, usrmsg],
        "gold_queries": query,
    }

#ds = load_dataset("json", data_files="data/train_small.jsonl", split="train")
ds = load_dataset("json", data_files="data/train_1.jsonl", split="train")
ds = ds.map(convert_ds, remove_columns=ds.column_names)


trainer = GRPOTrainer(
    model=MODEL,
    processing_class=AutoTokenizer.from_pretrained(MODEL),
    reward_funcs=[jena_reward],
    train_dataset=ds,
    args=GRPOConfig(num_generations=8,
                    generation_batch_size=8,
                    max_completion_length=256,
                    learning_rate=1e-6,                     # 1/10th of the LR for SFT.
                    warmup_steps=20,                         # FIXME: Increase with larger DS.
                    lr_scheduler_type="cosine",
                    bf16=True, fp16=False,
                    optim="adamw_torch_fused",
                    max_grad_norm=.5,
                    beta=.03,
                    temperature=1.2,
                    report_to="mlflow",
                    logging_steps=1, logging_first_step=True,
                    # Makes MLFlow run name always the same: seed=1234,
                    num_train_epochs=3,
                    prediction_loss_only=True,
                    use_vllm=False, vllm_gpu_memory_utilization=0.3),
)
trainer.train()

trainer.save_model(OUTPUT_DIR)
#mlflow.transformers.log_model(transformers_model={"model": trainer.model, "tokenizer": trainer.model.tokenizer_no_pad}, name="model", save_pretrained=False)
mlflow.log_artifact(__file__, artifact_path="code")
