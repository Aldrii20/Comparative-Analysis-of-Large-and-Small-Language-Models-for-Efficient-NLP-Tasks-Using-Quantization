from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_K_M.gguf",
    local_dir="green_nlp_project/models/quantized_llm",
    local_dir_use_symlinks=False
)

print(f"Model downloaded to: {model_path}")
