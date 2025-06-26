import os
import time
import json
from datasets import load_dataset
import evaluate

from codecarbon import EmissionsTracker
from llama_cpp import Llama

# === CONFIG ===
MODEL_PATH = r"green_nlp_project\models\quantized_llm\qa\model\llama-2-7b-32k-instruct.Q4_K_M.gguf"
RESULTS_PATH = r"green_nlp_project\evaluation\eval_results\quantized_llm\qa\qa_eval_with_energy.json"
MAX_SAMPLES = 100
MAX_TOKENS = 256
N_THREADS = 6
N_GPU_LAYERS = 35

# === Load model ===
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False
)

# === Load SQuAD dataset ===
dataset = load_dataset("squad", split=f"validation[:{MAX_SAMPLES}]")
metric = evaluate.load("squad")


# === Prompt template ===
def build_prompt(context, question):
    return f"""Answer the following question based on the context below.

Context:
{context}

Question:
{question}

Answer:"""

# === Start emissions tracking ===
tracker = EmissionsTracker(
    project_name="Quantized_LLaMA_QA_Eval",
    output_dir=os.path.dirname(RESULTS_PATH),
    output_file="qa_energy_llama2_7b_gguf.csv"
)
tracker.start()

start_time = time.time()
predictions = []
references = []

# === Inference Loop ===
for example in dataset:
    context = example["context"]
    question = example["question"]
    prompt = build_prompt(context, question)

    try:
        output = llm(prompt, max_tokens=MAX_TOKENS, stop=["\n"], echo=False)
        predicted_answer = output["choices"][0]["text"].strip()
    except Exception as e:
        predicted_answer = ""

    predictions.append({
        "id": example["id"],
        "prediction_text": predicted_answer
    })

    references.append({
        "id": example["id"],
        "answers": example["answers"]
    })

end_time = time.time()
tracker.stop()

# === Calculate metrics ===
eval_metrics = metric.compute(predictions=predictions, references=references)

latency_per_query_ms = ((end_time - start_time) * 1000) / MAX_SAMPLES
energy_kwh = tracker.final_emissions_data.energy_consumed
energy_joules = energy_kwh * 3.6e6
emissions_kg = tracker.final_emissions_data.emissions
energy_per_query = energy_joules / MAX_SAMPLES
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

# === Final Results ===
results = {
    "model": "llama-2-7b-32k-instruct.Q4_K_M.gguf",
    "exact_match": round(eval_metrics["exact_match"], 2),
    "f1": round(eval_metrics["f1"], 2),
    "latency_ms": round(latency_per_query_ms, 2),
    "energy_per_query_j": round(energy_per_query, 2),
    "total_energy_j": round(energy_joules, 2),
    "emissions_kg": round(emissions_kg, 6),
    "model_size_mb": round(model_size_mb, 2),
    "num_samples": MAX_SAMPLES
}

# === Save to JSON ===
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

# === Done ===
print("✅ Quantized QA Evaluation Complete.")
print(json.dumps(results, indent=4))
