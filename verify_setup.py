"""Quick smoke test for dataset loading and environment setup."""
import torch

from src.dataset.char_dataset import get_char_datasets
from src.dataset.line_dataset import get_line_dataloaders
from src.utils.device import configure_gpu


def main():
    configure_gpu()
    print("=== Stage 1: Dataset454 ===")
    train_loader, valid_loader, test_loader, classes = get_char_datasets(
        "Datasets/Dataset454", batch_size=4
    )
    print(f"Classes: {len(classes)}")
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, labels: {labels.shape}")

    print("\n=== Stage 2: SinOCR-handwritten + SinOCR-printed ===")
    train_loader, val_loader = get_line_dataloaders(batch_size=2)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    batch = next(iter(train_loader))
    print(f"pixel_values: {batch['pixel_values'].shape}")
    print(f"labels: {batch['labels'].shape}")

    print("\n=== PyTorch / CUDA ===")
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
