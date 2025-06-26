import os
import json
import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from codecarbon import EmissionsTracker
import evaluate

# === Configuration ===
MODEL_PATH = r"green_nlp_project/models/quantized_llm/summarization/model"
RESULTS_PATH = r"green_nlp_project/evaluation/eval_results/quantized_llm/summarization/summarization_eval_with_energy.json"
MODEL_NAME = os.path.basename(MODEL_PATH)
DATASET_NAME = "cnn_dailymail"
DATASET_CONFIG = "3.0.0"
MAX_SAMPLES = 100

# === Load tokenizer and quantized model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoGPTQForCausalLM.from_quantized(
    MODEL_PATH,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_safetensors=True,
    trust_remote_code=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# === Load dataset ===
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
val_data = dataset["validation"].shuffle(seed=42).select(range(MAX_SAMPLES))

# === Preprocess ===
def preprocess(article):
    return f"Summarize the following article:\n\n{article}\n\nSummary:"

# === ROUGE metric ===
rouge = evaluate.load("rouge")

# === CodeCarbon Tracker ===
tracker = EmissionsTracker(
    project_name="QuantizedLLM_Summarization_Eval",
    output_dir=os.path.dirname(RESULTS_PATH),
    output_file="quant_llm_summarization_energy.csv"
)
tracker.start()
start = time.time()

# === Evaluation Loop ===
decoded_preds = []
decoded_labels = []
acc_vals, precision_vals, recall_vals, f1_vals = [], [], [], []

for example in val_data:
    prompt = preprocess(example["article"])
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, max_new_tokens=128)

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    label = example["highlights"]

    decoded_preds.append(summary)
    decoded_labels.append(label)

    pred_tokens = set(summary.lower().split())
    label_tokens = set(label.lower().split())

    tp = len(pred_tokens & label_tokens)
    fp = len(pred_tokens - label_tokens)
    fn = len(label_tokens - pred_tokens)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = tp / (len(label_tokens) + 1e-8)

    precision_vals.append(precision)
    recall_vals.append(recall)
    f1_vals.append(f1)
    acc_vals.append(acc)

# === Metrics ===
rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)
end = time.time()
latency_per_query = ((end - start) * 1000) / MAX_SAMPLES

# === Energy and Emissions ===
tracker.stop()
energy_kwh = tracker.final_emissions_data.energy_consumed
emissions_kg = tracker.final_emissions_data.emissions
energy_joules = energy_kwh * 3.6e6
energy_per_query = energy_joules / MAX_SAMPLES

# === Model Size ===
model_size = sum(
    os.path.getsize(os.path.join(MODEL_PATH, f)) for f in os.listdir(MODEL_PATH)
) / (1024 * 1024)

# === Save Results ===
results = {
    "model": MODEL_NAME,
    "accuracy": round(np.mean(acc_vals) * 100, 2),
    "f1": round(np.mean(f1_vals) * 100, 2),
    "precision": round(np.mean(precision_vals) * 100, 2),
    "recall": round(np.mean(recall_vals) * 100, 2),
    "rougeL": round(rouge_score["rougeL"] * 100, 2),
    "latency_ms": round(latency_per_query, 2),
    "energy_per_query_j": round(energy_per_query, 4),
    "total_energy_j": round(energy_joules, 2),
    "emissions_kg": round(emissions_kg, 6),
    "model_size_mb": round(model_size, 2),
    "num_samples": MAX_SAMPLES
}

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("✅ Quantized LLM Summarization Evaluation Complete.")
print(json.dumps(results, indent=4))
