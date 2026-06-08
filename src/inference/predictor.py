import io
import os
from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor
from torchvision.datasets import ImageFolder

from src.models.trocr_model import SinhalaTrOCR
from src.models.vision_encoder import DeiTClassifier
from src.utils.device import configure_gpu


PROCESSOR_NAME = "google/vit-base-patch16-384"
TOKENIZER_NAME = "keshan/SinhalaBERTo"
STAGE1_WEIGHTS = "outputs/stage1/best_classifier.pth"
STAGE2_WEIGHTS = "outputs/stage2/best_trocr.pth"
CLASS_DATA_DIR = "Datasets/Dataset454/train"


class ModelService:
    def __init__(self):
        self.device, self.use_amp = configure_gpu()
        self.processor = ViTImageProcessor.from_pretrained(PROCESSOR_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self._stage1_model = None
        self._stage2_model = None
        self._class_names = None

    @property
    def class_names(self):
        if self._class_names is None and os.path.isdir(CLASS_DATA_DIR):
            dataset = ImageFolder(CLASS_DATA_DIR)
            self._class_names = dataset.classes
        return self._class_names or []

    def status(self):
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "stage1_ready": os.path.exists(STAGE1_WEIGHTS),
            "stage2_ready": os.path.exists(STAGE2_WEIGHTS),
            "num_classes": len(self.class_names),
        }

    def _load_stage1(self):
        if self._stage1_model is not None:
            return self._stage1_model

        if not os.path.exists(STAGE1_WEIGHTS):
            raise FileNotFoundError(
                f"Stage 1 weights not found at {STAGE1_WEIGHTS}. Run train_stage1.py first."
            )

        num_classes = len(self.class_names) or 454
        model = DeiTClassifier(num_classes=num_classes).to(self.device)
        model.load_state_dict(
            torch.load(STAGE1_WEIGHTS, map_location=self.device, weights_only=True)
        )
        model.eval()
        self._stage1_model = model
        return model

    def _load_stage2(self):
        if self._stage2_model is not None:
            return self._stage2_model

        encoder_path = (
            "outputs/stage1/pretrained_encoder.pth"
            if os.path.exists("outputs/stage1/pretrained_encoder.pth")
            else None
        )
        model = SinhalaTrOCR(encoder_pretrained_path=encoder_path).to(self.device)

        if os.path.exists(STAGE2_WEIGHTS):
            model.load_state_dict(
                torch.load(STAGE2_WEIGHTS, map_location=self.device, weights_only=True)
            )
        else:
            raise FileNotFoundError(
                f"Stage 2 weights not found at {STAGE2_WEIGHTS}. Run train_stage2.py first."
            )

        model.eval()
        self._stage2_model = model
        return model

    def _image_from_bytes(self, image_bytes):
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def _pixel_values(self, image):
        return self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

    def predict_character(self, image_bytes, top_k=5):
        image = self._image_from_bytes(image_bytes)
        model = self._load_stage1()
        pixel_values = self._pixel_values(image)

        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast(self.device.type):
                    logits = model(pixel_values)
            else:
                logits = model(pixel_values)

            probabilities = torch.softmax(logits, dim=-1)[0]
            top_scores, top_indices = torch.topk(probabilities, min(top_k, len(self.class_names)))

        predictions = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            label = self.class_names[idx] if idx < len(self.class_names) else str(idx)
            predictions.append(
                {
                    "label": label,
                    "confidence": round(score * 100, 2),
                }
            )

        return {
            "top_prediction": predictions[0],
            "predictions": predictions,
        }

    def predict_line(self, image_bytes):
        image = self._image_from_bytes(image_bytes)
        model = self._load_stage2()
        pixel_values = self._pixel_values(image)

        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast(self.device.type):
                    generated_ids = model.generate(pixel_values)
            else:
                generated_ids = model.generate(pixel_values)

        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return {"text": text}

    def predict_document(self, image_bytes):
        from src.preprocessing.document import extract_lines_from_bytes
        import base64

        extraction = extract_lines_from_bytes(image_bytes, return_debug=True)
        line_results = []

        for line in extraction["lines"]:
            image_data = base64.b64decode(line["image_base64"])
            prediction = self.predict_line(image_data)
            line_results.append(
                {
                    "index": line["index"],
                    "image_base64": line["image_base64"],
                    "text": prediction["text"],
                }
            )

        full_text = "\n".join(item["text"] for item in line_results if item["text"])

        return {
            "line_count": extraction["line_count"],
            "lines": line_results,
            "full_text": full_text,
            "debug": extraction.get("debug", {}),
        }


@lru_cache(maxsize=1)
def get_model_service():
    return ModelService()
