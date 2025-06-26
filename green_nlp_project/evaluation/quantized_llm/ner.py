import os
import time
import json
from datasets import load_dataset
from codecarbon import EmissionsTracker
from llama_cpp import Llama
from seqeval.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score

# === Config ===
MODEL_PATH = r"green_nlp_project\models\quantized_llm\ner\model\llama-2-7b-chat.Q4_K_M.gguf"
RESULTS_PATH = r"green_nlp_project\evaluation\eval_results\quantized_llm\ner\ner_energy.json"
MAX_SAMPLES = 200
MAX_TOKENS = 256

# === Load model ===
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_threads=6,
    n_gpu_layers=35,
    verbose=False
)

# === Load dataset ===
dataset = load_dataset("wnut_17", split=f"validation[:{MAX_SAMPLES}]")
label_list = dataset.features["ner_tags"].feature.names

# === Prompt Template with few-shot example ===
def build_prompt(tokens):
    sentence = " ".join(tokens)
    example = (
        "Input: Elon Musk founded SpaceX.\n"
        "Output:\n"
        "Elon B-PER\n"
        "Musk I-PER\n"
        "SpaceX B-ORG\n\n"
    )
    return (
        "You are an NER tagger. For each word in the input, return the token and its NER tag, separated by space.\n"
        "Use BIO format (B-PER, I-ORG, O, etc).\n"
        "Example:\n"
        f"{example}"
        f"Input: {sentence}\n"
        "Output:\n"
    )

# === Track energy ===
tracker = EmissionsTracker(
    project_name="Quantized_LLaMA_NER_GGUF",
    output_dir=os.path.dirname(RESULTS_PATH),
    output_file="llama2_gguf_ner_energy.csv"
)
tracker.start()

start_time = time.time()
true_labels = []
pred_labels = []

for example in dataset:
    tokens = example["tokens"]
    gold_tags = [label_list[i] for i in example["ner_tags"]]
    prompt = build_prompt(tokens)

    output = llm(prompt, max_tokens=MAX_TOKENS, stop=["\n\n", "Input:"], echo=False)
    response = output["choices"][0]["text"].strip()

    # Parse response into predicted tags
    pred_tags = []
    for line in response.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            token, tag = parts
            if tag in label_list:
                pred_tags.append(tag)
            else:
                pred_tags.append("O")
        else:
            pred_tags.append("O")

    # Ensure same length
    if len(pred_tags) != len(gold_tags):
        pred_tags = ["O"] * len(gold_tags)

    true_labels.append(gold_tags)
    pred_labels.append(pred_tags)

end_time = time.time()
tracker.stop()

# === Metrics ===
latency_ms = ((end_time - start_time) * 1000) / MAX_SAMPLES
energy_joules = tracker.final_emissions_data.energy_consumed * 3.6e6
emissions_kg = tracker.final_emissions_data.emissions
energy_per_query = energy_joules / MAX_SAMPLES

model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)

results = {
    "model": "llama2_7b_chat_q4_gguf",
    "accuracy": round(accuracy_score(true_labels, pred_labels), 4),
    "f1": round(f1_score(true_labels, pred_labels), 4),
    "precision": round(precision_score(true_labels, pred_labels), 4),
    "recall": round(recall_score(true_labels, pred_labels), 4),
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

print("✅ Quantized LLaMA NER Evaluation Complete.")
print(json.dumps(results, indent=4))
