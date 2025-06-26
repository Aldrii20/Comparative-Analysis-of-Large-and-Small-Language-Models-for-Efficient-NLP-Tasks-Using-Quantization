from datasets import load_dataset

# Download and cache dataset into the 'datasets/downloaded/' directory
dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir="datasets/downloaded/")
