import asyncio, re, aiohttp, torch
from statistics import mean, stdev
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import time
from torch.optim import Muon
import mlflow

from yago_prompts import SYSTEM_PROMPT
from validate import strip_reasoning

FUSEKI = "http://192.168.8.110:3030/yago/sparql"
#MODEL = "./output/qwen3.5-0.8b-sparql/checkpoint-294/"
MODEL = "./output/qwen3.5-4b-sparql/checkpoint-292/" # FIXME: 4B
OUTPUT_DIR = "output/qwen3.5-4b-sparql-rl"

# MLflow
#MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"   # local, no server needed
MLFLOW_TRACKING_URI = "http://192.168.8.110:5000/"
MLFLOW_EXPERIMENT = "qwen3.5-4b-sparql-rl"

# ---------------------------------------------------------------------------
# MLflow run: log config, train, log artifacts + a quick parse-rate check
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)


BASE_MODEL_NAME = "Qwen/Qwen3.5-4B"        # or Qwen/Qwen3-1.7B for max speed
#MODEL_NAME = "Qwen/Qwen3.5-0.8B"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME, torch_dtype=torch.bfloat16
).to("xpu")
model = PeftModel.from_pretrained(model, MODEL, is_trainable=True)

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

async def jena_reward(completions, gold_queries, log_metric, **kwargs):
    valid_ratios = []
    num_results_correct = []
    num_results_expected = []
    correct_ratios = []
    query_durations_s = []
    async def score(completion, gold_query):
        if completion is None:
            return -0.5

        gold = await execute_sparql(gold_query["content"]) # Get actual expected result.

        start_time = time()
        query, reasoning = strip_reasoning(completion[0]["content"])
        pred = await execute_sparql(query)

        duration_s = time() - start_time
        query_durations_s.append(duration_s)

        reasoning_factor = 8192. / (8192. + len(reasoning))          # In chars, not in tokens.

        if pred is None:
            # Query failed to run.
            valid_ratios.append(0.)
            num_results_correct.append(0.)
            num_results_expected.append(len(gold))
            correct_ratios.append(0.)
            return -0.1 - 0.1*reasoning_factor

        nc = len(pred & gold)
        er = len(pred | gold)
        correct_ratio = nc / (er + 1e-7)
        valid_ratios.append(1.)
        num_results_correct.append(nc)
        num_results_expected.append(er)
        correct_ratios.append(correct_ratio)

        if not pred and not gold:
            # No results, intended.
            return 0.5 * reasoning_factor

        if not pred:
            # No results, not intended.
            return 0.1 * reasoning_factor

        if gold is None or not gold:
            # Nothing to compare to.
            return 0.0

        #return (0.2 + 0.8 * len(pred & gold) / len(pred | gold)) * ((3. / (3. + duration_s)) ** 2)
        return (0.2 + 0.8 * correct_ratio) * (1. + reasoning_factor)

    result = await asyncio.gather(*(score(c, g) for c, g in zip(completions, gold_queries)))

    log_metric("mean_valid_ratio", mean(valid_ratios))
    log_metric("std_valid_ratio", stdev(valid_ratios))
    log_metric("mean_num_results_correct", mean(num_results_correct))
    log_metric("std_num_results_correct", stdev(num_results_correct))
    log_metric("mean_num_results_expected", mean(num_results_expected))
    log_metric("std_num_results_expected", stdev(num_results_expected))
    log_metric("mean_correct_ratio", mean(correct_ratios))
    log_metric("std_correct_ratio", stdev(correct_ratios))
    log_metric("mean_query_duration_s", mean(query_durations_s))
    log_metric("std_query_duration_s", stdev(query_durations_s))

    return result

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
ds = load_dataset("json", data_files="data/train_rl.jsonl", split="train")
ds = ds.map(convert_ds, remove_columns=ds.column_names)

#optimizer = Muon((p for p in model.parameters() if p.requires_grad), adjust_lr_fn='match_rms_adamw')    # LR is handled by GRPOConfig.

trainer = GRPOTrainer(
    model=model,
    processing_class=AutoTokenizer.from_pretrained(MODEL),
    reward_funcs=[jena_reward],
    train_dataset=ds,
    #optimizers=(optimizer, None),
    args=GRPOConfig(num_generations=8,
                    generation_batch_size=8,
                    per_device_train_batch_size=8,
                    gradient_accumulation_steps=4,
                    max_completion_length=256,
                    #learning_rate=3e-6,                     # 1/10th of the LR for SFT.
                    learning_rate=5e-5,                     # 1/10th of the LR for SFT.  Higher for LoRA.
                    warmup_steps=10,                         # FIXME: Increase with larger DS.
                    lr_scheduler_type="cosine",
                    bf16=True, fp16=False,
                    optim="adamw_torch_fused",
                    #optim="adamw_torch",
                    #max_grad_norm=1.0,
                    max_grad_norm=0.2,
                    beta=.02,
                    temperature=0.9,                        # Above 1 produces pure garbage with mangled reasoning traces.
                    report_to="mlflow",
                    logging_steps=1, logging_first_step=True,
                    # Makes MLFlow run name always the same: seed=1234,
                    num_train_epochs=1,
                    prediction_loss_only=True,
                    mask_truncated_completions=False,
                    save_strategy='steps',
                    save_steps=50,
                    save_total_limit=3,
                    save_only_model=True,
                    use_vllm=False, vllm_gpu_memory_utilization=0.3),
)
trainer.train()

trainer.save_model(OUTPUT_DIR)
#mlflow.transformers.log_model(transformers_model={"model": trainer.model, "tokenizer": trainer.model.tokenizer_no_pad}, name="model", save_pretrained=False)
mlflow.log_artifact(__file__, artifact_path="code")
