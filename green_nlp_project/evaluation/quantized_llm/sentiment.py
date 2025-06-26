import os
import time
import json
import torch
from datasets import load_dataset
from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === Paths ===
MODEL_DIR = r"green_nlp_project\models\quantized_llm\sentiment\model"
RESULTS_PATH = r"green_nlp_project\evaluation\eval_results\quantized_llm\sentiment\sentiment_eval_with_energy.json"
MAX_SAMPLES = 200

# === Load model + tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(
    MODEL_DIR,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    low_cpu_mem_usage=True,
    use_safetensors=True,
    trust_remote_code=True
)

# === Load dataset (SST-2) ===
dataset = load_dataset("glue", "sst2", split=f"validation[:{MAX_SAMPLES}]")

# === Build prompt ===
def build_prompt(sentence):
    return f"""Classify the sentiment of the sentence as Positive or Negative.

Sentence: "{sentence}"
Sentiment:"""

# === Inference pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20)

# === Start CodeCarbon ===
tracker = EmissionsTracker(
    project_name="Quantized_LLaMA_GPTQ_Sentiment",
    output_dir=os.path.dirname(RESULTS_PATH),
    output_file="llama_gptq_sentiment_energy.csv"
)
tracker.start()
start = time.time()

# === Evaluation ===
true_labels = []
pred_labels = []

for example in dataset:
    sentence = example["sentence"]
    true_label = example["label"]

    prompt = build_prompt(sentence)
    response = generator(prompt)[0]["generated_text"].split("Sentiment:")[-1].strip().lower()

    if "positive" in response:
        pred = 1
    elif "negative" in response:
        pred = 0
    else:
        pred = 1  # fallback

    true_labels.append(true_label)
    pred_labels.append(pred)

end = time.time()
tracker.stop()

# === Metrics ===
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)
latency_ms = (end - start) * 1000 / MAX_SAMPLES
energy_joules = tracker.final_emissions_data.energy_consumed * 3.6e6
emissions_kg = tracker.final_emissions_data.emissions
energy_per_query = energy_joules / MAX_SAMPLES
model_size = sum(os.path.getsize(os.path.join(MODEL_DIR, f)) for f in os.listdir(MODEL_DIR)) / (1024 * 1024)

# === Save results ===
results = {
    "model": os.path.basename(MODEL_DIR),
    "accuracy": round(accuracy * 100, 2),
    "f1": round(f1 * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "latency_ms": round(latency_ms, 2),
    "energy_per_query_j": round(energy_per_query, 4),
    "total_energy_j": round(energy_joules, 2),
    "emissions_kg": round(emissions_kg, 6),
    "model_size_mb": round(model_size, 2),
    "num_samples": MAX_SAMPLES
}

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("✅ Sentiment Evaluation (GPTQ) Complete.")
print(json.dumps(results, indent=4))
