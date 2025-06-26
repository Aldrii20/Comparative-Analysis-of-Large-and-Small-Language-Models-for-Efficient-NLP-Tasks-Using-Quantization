from datasets import load_dataset
import os

def save_dataset(dataset, save_path):
    print(f"Saving to: {save_path}")
    dataset.save_to_disk(save_path)

def download_all_datasets():
    base_path = os.path.join("datasets", "downloaded")
    os.makedirs(base_path, exist_ok=True)

    # 1. SST-2: Sentiment Analysis
    print("Downloading SST-2...")
    sst2 = load_dataset("glue", "sst2")
    save_dataset(sst2, os.path.join(base_path, "sst2"))

    # 2. WNUT-17: Named Entity Recognition
    print("Downloading WNUT-17...")
    wnut = load_dataset("wnut_17")
    save_dataset(wnut, os.path.join(base_path, "wnut_17"))

   

    # 4. SQuAD: Question Answering
    print("Downloading SQuAD...")
    squad = load_dataset("squad")
    save_dataset(squad, os.path.join(base_path, "squad"))

    print("\n✅ All datasets downloaded and saved.")

if __name__ == "__main__":
    download_all_datasets()
