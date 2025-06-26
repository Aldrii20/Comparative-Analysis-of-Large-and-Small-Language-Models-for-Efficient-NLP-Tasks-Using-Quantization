import os
import json
import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from codecarbon import EmissionsTracker
import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ✅ Configuration
MODEL_DIR = r"green_nlp_project\results\slm\summarization\flan-t5"
RESULTS_PATH = r"green_nlp_project/evaluation/eval_results/slm/summarization/summarization_eval_with_energy.json"
MODEL_NAME = os.path.basename(MODEL_DIR)
DATASET_NAME = "cnn_dailymail"
DATASET_CONFIG = "3.0.0"

# ✅ Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Load dataset
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
val_data = dataset["validation"].shuffle(seed=42).select(range(1000))

# ✅ Preprocessing
def preprocess_function(examples, max_input_length=512, max_target_length=128):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_val = val_data.map(preprocess_function, batched=True, remove_columns=val_data.column_names)

# ✅ Load ROUGE metric
rouge = evaluate.load("rouge")

# ✅ Custom metric logic
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE
    rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    # Convert to token sets
    pred_token_lists = [set(pred.lower().split()) for pred in decoded_preds]
    label_token_lists = [set(label.lower().split()) for label in decoded_labels]

    precision_vals = []
    recall_vals = []
    f1_vals = []
    acc_vals = []

    for pred_tokens, label_tokens in zip(pred_token_lists, label_token_lists):
        if not label_tokens:
            continue

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

    return {
        "rougeL": rouge_scores["rougeL"],
        "accuracy": np.mean(acc_vals),
        "precision": np.mean(precision_vals),
        "recall": np.mean(recall_vals),
        "f1": np.mean(f1_vals),
    }

# ✅ Start CodeCarbon
tracker = EmissionsTracker(
    project_name="SLM_Summarization_Eval",
    output_dir=os.path.dirname(RESULTS_PATH),
    output_file="slm_summarization_energy.csv"
)
tracker.start()

# ✅ Timer
start = time.time()

# ✅ Evaluation
training_args = Seq2SeqTrainingArguments(
    output_dir="tmp_output_summarization_eval",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

metrics = trainer.evaluate()

end = time.time()
latency_per_query = (end - start) * 1000 / len(tokenized_val)

# ✅ Stop CodeCarbon
tracker.stop()
energy_kwh = tracker.final_emissions_data.energy_consumed
emissions_kg = tracker.final_emissions_data.emissions
energy_joules = energy_kwh * 3.6e6
energy_per_query = energy_joules / len(tokenized_val)

# ✅ Model size
model_size = sum(
    os.path.getsize(os.path.join(MODEL_DIR, f)) for f in os.listdir(MODEL_DIR)
) / (1024 * 1024)

# ✅ Save final results
results = {
    "model": MODEL_NAME,
    "accuracy": round(metrics["eval_accuracy"] * 100, 2),
    "f1": round(metrics["eval_f1"] * 100, 2),
    "precision": round(metrics["eval_precision"] * 100, 2),
    "recall": round(metrics["eval_recall"] * 100, 2),
    "rougeL": round(metrics["eval_rougeL"] * 100, 2),
    "latency_ms": round(latency_per_query, 2),
    "energy_per_query_j": round(energy_per_query, 4),
    "total_energy_j": round(energy_joules, 2),
    "emissions_kg": round(emissions_kg, 6),
    "model_size_mb": round(model_size, 2),
    "num_samples": len(tokenized_val)
}

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("✅ SLM Summarization Evaluation Complete.")
print(json.dumps(results, indent=4))
