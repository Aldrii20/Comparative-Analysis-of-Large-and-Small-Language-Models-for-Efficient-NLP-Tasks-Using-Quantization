import os

# Set your base directory
base_path = r"D:\Research Project\Green NLP\Comparative-Analysis-of-Large-and-Small-Language-Models-for-Efficient-NLP-Tasks-Using-Quantization"

# Define the relative folder structure
folders = [
    "data/raw",
    "data/processed",
    "models/slm",
    "models/llm",
    "models/llm_quantized",
    "tasks",
    "evaluation",
    "utils",
    "notebooks",
    "results"
]

# Create folders
for folder in folders:
    path = os.path.join(base_path, folder)
    os.makedirs(path, exist_ok=True)
    print(f"✅ Created: {path}")

# Define placeholder files (optional)
files = [
    "requirements.txt",
    "run_pipeline.py",
    "README.md",
    "utils/data_loader.py",
    "utils/model_loader.py",
    "evaluation/metrics.py",
    "evaluation/resource_monitor.py",
    "tasks/sentiment_analysis.py",
    "tasks/question_answering.py",
    "tasks/named_entity_recognition.py",
    "tasks/text_summarization.py"
]

# Create empty files
for file in files:
    path = os.path.join(base_path, file)
    with open(path, "w") as f:
        pass
    print(f"📝 Created empty file: {path}")
