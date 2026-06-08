import os

import torch


def configure_gpu():
    """Enable maximum GPU throughput for training."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — training will run on CPU.")
        print("Run install_gpu.ps1 to install PyTorch with CUDA 12.8 (required for RTX 5090).")
        return torch.device("cpu"), False

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB VRAM)")
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    return device, True


def dataloader_kwargs(num_workers=None, pin_memory=True):
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 4)

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs
