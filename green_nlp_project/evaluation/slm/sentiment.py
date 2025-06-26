import os
import json
import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, default_data_collator
from codecarbon import EmissionsTracker

# === Configuration ===
MODEL_DIR = r"green_nlp_project\results\slm\sentiment\checkpoint-4210"
RESULTS_PATH = r"green_nlp_project\evaluation\eval_results\slm\seniment\sentiment_metrics.json"
MODEL_NAME = os.path.basename(MODEL_DIR)
DATASET_NAME = "sst2"

# === Load tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Load dataset ===
dataset = load_dataset("glue", DATASET_NAME)
tokenized_dataset = dataset.map(lambda x: tokenizer(x["sentence"], padding="max_length", truncation=True), batched=True)
tokenized_dataset.set_format("torch")

# === Metric function ===
def compute_metrics(p): 
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    accuracy = np.mean(preds == labels)
    return {"accuracy": accuracy}

# === Start CodeCarbon ===
tracker = EmissionsTracker(
    project_name="SLM_Sentiment_Eval",
    output_dir=os.path.dirname(RESULTS_PATH),
    output_file="slm_sentiment_energy.csv"
)
tracker.start()

# === Timer for latency ===
start = time.time()

# === Trainer and evaluation ===
training_args = TrainingArguments(
    output_dir="tmp_output_sentiment_eval",
    per_device_eval_batch_size=8,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)

metrics = trainer.evaluate()

end = time.time()
latency_per_query = (end - start) * 1000 / len(tokenized_dataset["validation"])  # ms

# === Stop CodeCarbon ===
tracker.stop()
energy_kwh = tracker.final_emissions_data.energy_consumed
emissions_kg = tracker.final_emissions_data.emissions
energy_joules = energy_kwh * 3.6e6
energy_per_query = energy_joules / len(tokenized_dataset["validation"])

# === Estimate model size (MB) ===
model_size = sum(
    os.path.getsize(os.path.join(MODEL_DIR, f)) for f in os.listdir(MODEL_DIR)
) / (1024 * 1024)

# === Save metrics ===
results = {
    "model": MODEL_NAME,
    "accuracy": round(metrics["eval_accuracy"] * 100, 2),
    "latency_ms": round(latency_per_query, 2),
    "energy_per_query_j": round(energy_per_query, 4),
    "total_energy_j": round(energy_joules, 2),
    "emissions_kg": round(emissions_kg, 6),
    "model_size_mb": round(model_size, 2),
    "num_samples": len(tokenized_dataset["validation"])
}

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("✅ SLM Sentiment Evaluation Complete.")
print(json.dumps(results, indent=4))
