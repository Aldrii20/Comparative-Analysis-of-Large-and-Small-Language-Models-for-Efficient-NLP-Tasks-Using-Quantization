import os
import json
import time
import torch
import numpy as np
from datasets import load_from_disk
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from codecarbon import EmissionsTracker

# === Configuration ===
MODEL_DIR = r"green_nlp_project/results/slm/ner"
DATASET_DIR = r"datasets/downloaded/wnut_17"
RESULTS_PATH = r"green_nlp_project/evaluation/eval_results/slm/ner.json"  # ✅ fixed: point to file
MODEL_NAME = os.path.basename(MODEL_DIR)

# === Load tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Load dataset ===
dataset = load_from_disk(DATASET_DIR)
label_names = dataset["train"].features["ner_tags"].feature.names

# === Tokenization ===
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
tokenized_dataset.set_format("torch")

# === Metrics ===
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=2)
    labels = p.label_ids

    true_preds = [
        [label_names[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "accuracy": results["overall_accuracy"],
        "f1": results["overall_f1"],
        "precision": results["overall_precision"],
        "recall": results["overall_recall"]
    }

# === Start CodeCarbon ===
tracker = EmissionsTracker(
    project_name="SLM_NER_Eval",
    output_dir=os.path.dirname(RESULTS_PATH),
    output_file="slm_ner_energy.csv"
)
tracker.start()

# === Evaluation ===
start = time.time()

args = TrainingArguments(
    output_dir="tmp_output",
    per_device_eval_batch_size=16,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    eval_dataset=tokenized_dataset["validation"]
)

metrics = trainer.evaluate()

end = time.time()
latency_per_query = (end - start) * 1000 / len(tokenized_dataset["validation"])  # in ms

# === Stop CodeCarbon ===
tracker.stop()
energy_kwh = tracker.final_emissions_data.energy_consumed
emissions_kg = tracker.final_emissions_data.emissions
energy_joules = energy_kwh * 3.6e6
energy_per_query = energy_joules / len(tokenized_dataset["validation"])

# === Model Size (MB) ===
model_size_mb = sum(
    os.path.getsize(os.path.join(MODEL_DIR, f)) for f in os.listdir(MODEL_DIR)
) / (1024 * 1024)

# === Final Metrics ===
results = {
    "model": MODEL_NAME,
    "accuracy": round(metrics["eval_accuracy"] * 100, 2),
    "f1": round(metrics["eval_f1"] * 100, 2),
    "precision": round(metrics["eval_precision"] * 100, 2),
    "recall": round(metrics["eval_recall"] * 100, 2),
    "latency_ms": round(latency_per_query, 2),
    "energy_per_query_j": round(energy_per_query, 4),
    "total_energy_j": round(energy_joules, 2),
    "emissions_kg": round(emissions_kg, 6),
    "model_size_mb": round(model_size_mb, 2),
    "num_samples": len(tokenized_dataset["validation"])
}

# === Save Results ===
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("✅ SLM NER Evaluation Complete.")
print(json.dumps(results, indent=4))
