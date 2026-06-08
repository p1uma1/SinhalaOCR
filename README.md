## Setup

```powershell
cd D:\SinhalaOCR
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# GPU (required for RTX 5090 — needs CUDA 12.8, not default CPU build)
.\install_gpu.ps1
python verify_setup.py
```

**RTX 5090 note:** The default `pip install torch` gives a CPU-only build. Blackwell GPUs (sm_120) require PyTorch built with **CUDA 12.8** (`cu128`). Run `install_gpu.ps1` after creating the venv.

## Datasets

| Stage | Dataset | Path |
|-------|---------|------|
| Stage 1 | Dataset454 (454 Sinhala characters) | `Datasets/Dataset454/{train,valid,test}` |
| Stage 2 | SinOCR-handwritten (line OCR) | `Datasets/SinOCR-handwritten/handwritten-data/` |
| Stage 2 | SinOCR-printed (line OCR) | `Datasets/SinOCR-printed/data/` |

## Training

**Stage 1** — character classifier (ViT encoder + 454-class head):

```powershell
python train_stage1.py
```

Pipeline: `image → vision encoder → linear head (454 classes)`

Outputs: `outputs/stage1/best_classifier.pth`, `outputs/stage1/pretrained_encoder.pth`

**Stage 2** — Sinhala line OCR (TrOCR with SinBERT decoder):

```powershell
python train_stage2.py
```

Uses both SinOCR-handwritten and SinOCR-printed train splits; validates on their test splits. Loads the Stage 1 encoder from `outputs/stage1/pretrained_encoder.pth` when available.

Pipeline: `image → pretrained encoder → decoder → Sinhala word / sequence`

Output: `outputs/stage2/best_trocr.pth`

## Web App (Production Frontend + API)

Production React frontend with FastAPI backend.

### Production (public deploy)

```powershell
pip install -r requirements.txt
.\build_frontend.ps1
.\run_app.ps1
```

Open **http://127.0.0.1:8000**

### Development (hot reload)

```powershell
.\run_dev.ps1
```

Frontend: **http://127.0.0.1:5173** · API: **http://127.0.0.1:8000**

| Mode | Input | Model |
|------|-------|-------|
| **Line OCR** | One line image | Stage 2 TrOCR |
| **Document OCR** | Full page image | Preprocessing + Stage 2 per line |
| **Character** | Single character crop | Stage 1 classifier |

API docs: **http://127.0.0.1:8000/api/docs**

### Docker

```bash
docker build -t sinhala-ocr .
docker run -p 8000:8000 --gpus all sinhala-ocr
```

Copy `.env.example` to `.env` and set `ALLOWED_ORIGINS` for your public domain.