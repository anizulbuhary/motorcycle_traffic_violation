import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")
print(f"CUDA version in PyTorch: {torch.version.cuda}")
print(f"Is CUDA available: {torch.cuda.is_available()}")


