import torch

if torch.cuda.is_available():
    print("CUDA is available.")
    print("CUDA Version:", torch.version.cuda)
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Device Count:", torch.cuda.device_count())
else:
    print("CUDA is not available.")
