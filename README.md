python -m venv .venv
pip install -r requirements.txt

Calibrate the vision encoder first

Stage 1 — Build a character classifier

train:

image → vision encoder → linear head(454 classes)

DeiT models are designed for image classification, so adding a classification head on top of the encoder.

Stage 2 — Transfer the encoder

After training, keep only the encoder weights and load them into OCR model:

image → pretrained encoder → decoder → Sinhala word / sequence