import platform
import sys

import torch
import torchvision


def main() -> None:
    print("=== System ===")
    print(f"Python:      {sys.version.split()[0]}")
    print(f"Platform:    {platform.platform()}")
    print()

    print("=== Deep Learning Stack ===")
    print(f"PyTorch:     {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"CUDA:        {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA ver:    {torch.version.cuda}")
        print(f"GPU:         {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
