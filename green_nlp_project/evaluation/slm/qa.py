import os
import json
import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from codecarbon import EmissionsTracker

# === Configuration ===
MODEL_DIR = r"green_nlp_project/results/slm/qa"
RESULTS_PATH = r"green_nlp_project\evaluation\eval_results\slm\qa\qa_eval_with_energy.json"
MODEL_NAME = os.path.basename(MODEL_DIR)
DATASET_NAME = "squad"

# === Load tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Load and preprocess dataset ===
dataset = load_dataset(DATASET_NAME)

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]

    inputs = tokenizer(
        questions,
        contexts,
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
        return_token_type_ids=True,
    )

    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        answer = answers[i]["text"][0]
        start_char = answers[i]["answer_start"][0]
        end_char = start_char + len(answer)

        sequence_ids = inputs.sequence_ids(i)

        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        token_start_index = context_start
        token_end_index = context_end

        while token_start_index < context_end and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        while token_end_index > context_start and offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# === Metric function ===
def compute_metrics(p):
    start_preds = np.argmax(p.predictions[0], axis=1)
    end_preds = np.argmax(p.predictions[1], axis=1)
    start_labels = p.label_ids[0]
    end_labels = p.label_ids[1]
    accuracy = np.mean((start_preds == start_labels) & (end_preds == end_labels))
    return {"accuracy": accuracy}

# === Start CodeCarbon ===
tracker = EmissionsTracker(
    project_name="SLM_QA_Eval",
    output_dir=os.path.dirname(RESULTS_PATH),
    output_file="slm_qa_energy.csv"
)
tracker.start()

# === Timer for latency ===
start = time.time()

# === Trainer and evaluation ===
training_args = TrainingArguments(
    output_dir="tmp_output_qa_eval",
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
latency_per_query = (end - start) * 1000 / len(tokenized_dataset["validation"])  # in ms

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

print("✅ SLM QA Evaluation Complete.")
print(json.dumps(results, indent=4))
