import os

BASE_DIR = "Comparative-Analysis-of-Large-and-Small-Language-Models-for-Efficient-NLP-Tasks-Using-Quantization"
PROJECT_DIR = os.path.join(BASE_DIR, "green_nlp_project")

# Define folder structure
folders = [
    "configs/slm",
    "configs/llm",
    "configs/quantized",
    "datasets/downloaded",
    "models/slm",
    "models/llm",
    "models/quantized",
    "experiments/sentiment_analysis",
    "experiments/question_answering",
    "experiments/named_entity_recognition",
    "experiments/summarization",
    "evaluation",
    "results",
    "utils",
    "visualizations"
]

# Additional files to create
files = {
    "requirements.txt": os.path.join(PROJECT_DIR, "requirements.txt"),
    "README.md": os.path.join(PROJECT_DIR, "README.md"),
    ".gitignore": os.path.join(BASE_DIR, ".gitignore")
}

def create_folders():
    print("\n📁 Creating folder structure...")
    for folder in folders:
        path = os.path.join(PROJECT_DIR, folder)
        os.makedirs(path, exist_ok=True)
        print(f"✅ Created: {path}")

def create_files():
    print("\n📄 Creating base files...")

    # requirements.txt
    with open(files["requirements.txt"], "w") as f:
        f.write(
            "transformers\n"
            "datasets\n"
            "torch\n"
            "tqdm\n"
            "accelerate\n"
            "scikit-learn\n"
            "evaluate\n"
            "matplotlib\n"
        )
        print(f"✅ Created: {files['requirements.txt']}")

    # README.md
    with open(files["README.md"], "w") as f:
        f.write(
            "# Green NLP Project: SLM vs LLM vs Quantized LLM\n\n"
            "This repository compares SLMs, LLMs, and Quantized LLMs for core NLP tasks.\n"
            "It aims to optimize accuracy, latency, and energy use in low-resource environments.\n"
        )
        print(f"✅ Created: {files['README.md']}")

    # .gitignore
    with open(files[".gitignore"], "w") as f:
        f.write(
            "__pycache__/\n"
            "*.pyc\n"
            "*.ipynb_checkpoints/\n"
            "datasets/downloaded/\n"
            "models/\n"
            "*.log\n"
        )
        print(f"✅ Created: {files['.gitignore']}")

def main():
    create_folders()
    create_files()
    print("\n🎉 Project structure setup complete!")

if __name__ == "__main__":
    main()
