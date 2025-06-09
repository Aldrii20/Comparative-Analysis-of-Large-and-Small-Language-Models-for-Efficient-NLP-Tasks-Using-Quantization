import os

# Define your absolute base directory
base_dir = r"D:\Research Project\Green NLP\Comparative-Analysis-of-Large-and-Small-Language-Models-for-Efficient-NLP-Tasks-Using-Quantization"

# Subdirectories to create inside base_dir
folders = [
    "data/raw",
    "data/processed",
    "models/slm",
    "models/llm_quantized",
    "notebooks",
    "scripts",
    "results/logs",
    "results/figures"
]

# Create directories
for folder in folders:
    path = os.path.join(base_dir, folder)
    os.makedirs(path, exist_ok=True)

# Create optional files
for filename in ["README.md", "environment.yml"]:
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        open(file_path, "w").close()

print("✅ Full project structure created at:\n", base_dir)