import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === File Paths ===
slm_files = {
    "NER": r"green_nlp_project\evaluation\eval_results\slm\ner\ner.json",
    "QA": r"green_nlp_project\evaluation\eval_results\slm\qa\qa.json",
    "Sentiment": r"green_nlp_project\evaluation\eval_results\slm\sentiment\sentiment_metrics.json"
}

# === Load Quantized LLM JSON Files ===
llm_files = {
    "NER": r"green_nlp_project/evaluation/eval_results/quantized_llm/ner/ner_energy.json",
    "QA": r"green_nlp_project\evaluation\eval_results\quantized_llm\qa\qa_eval_with_energy.json",
    "Sentiment": r"green_nlp_project\evaluation\eval_results\quantized_llm\sentiment\sentiment_eval_with_energy.json"
}

metrics_to_plot = ["accuracy", "latency_ms", "energy_per_query_j"]

# === Load Data ===
data = []
for task, path in slm_files.items():
    with open(path) as f:
        val = json.load(f)
        data.append({"Model": "SLM", "Task": task, **{m: val.get(m, 0) for m in metrics_to_plot}})

for task, path in llm_files.items():
    with open(path) as f:
        val = json.load(f)
        data.append({"Model": "Quantized LLM", "Task": task, **{m: val.get(m, 0) for m in metrics_to_plot}})

df = pd.DataFrame(data)

# === Plot ===
sns.set(style="whitegrid")
fig, axs = plt.subplots(3, 3, figsize=(16, 10))

for row_idx, task in enumerate(df["Task"].unique()):
    task_df = df[df["Task"] == task]
    for col_idx, metric in enumerate(metrics_to_plot):
        ax = axs[row_idx, col_idx]
        sns.barplot(data=task_df, x="Model", y=metric, ax=ax, palette="Set2")
        ax.set_title(f"{task} - {metric.replace('_', ' ').title()}")
        ax.set_ylabel(metric if metric != "accuracy" else "Accuracy (%)")
        ax.set_xlabel("")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3)

plt.tight_layout()
plt.show()
