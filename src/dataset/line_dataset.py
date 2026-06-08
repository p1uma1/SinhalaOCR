import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import AutoTokenizer, ViTImageProcessor

# SinOCR dataset paths (relative to project root)
SINOCR_HANDWRITTEN = {
    "train": {
        "images_dir": "Datasets/SinOCR-handwritten/handwritten-data/train/images",
        "csv_file": "Datasets/SinOCR-handwritten/handwritten-data/train/data.csv",
    },
    "test": {
        "images_dir": "Datasets/SinOCR-handwritten/handwritten-data/test/images",
        "csv_file": "Datasets/SinOCR-handwritten/handwritten-data/test/data.csv",
    },
}

SINOCR_PRINTED = {
    "train": {
        "images_dir": "Datasets/SinOCR-printed/data/train/images",
        "csv_file": "Datasets/SinOCR-printed/data/train/gt.csv",
    },
    "test": {
        "images_dir": "Datasets/SinOCR-printed/data/test/images",
        "csv_file": "Datasets/SinOCR-printed/data/test/gt.csv",
    },
}


class LineDataset(Dataset):
    def __init__(
        self,
        images_dir,
        csv_file,
        processor=None,
        tokenizer=None,
        max_target_length=128,
        processor_name="google/vit-base-patch16-384",
        tokenizer_name="keshan/SinhalaBERTo",
    ):
        self.images_dir = images_dir
        self.processor = processor or ViTImageProcessor.from_pretrained(processor_name)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_target_length = max_target_length

        df = pd.read_csv(csv_file)
        self.samples = []
        for _, row in df.iterrows():
            file_name = str(row["file_name"])
            text = str(row["text"])
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                file_name = f"{file_name}.png"
            img_path = os.path.join(images_dir, file_name)
            if os.path.exists(img_path):
                self.samples.append((img_path, text))

        if not self.samples:
            raise FileNotFoundError(
                f"No valid image/label pairs found for {csv_file} in {images_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]

        labels = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def _collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def get_line_dataloader(
    images_dir,
    csv_file,
    batch_size=4,
    shuffle=True,
    num_workers=None,
    pin_memory=True,
    processor=None,
    tokenizer=None,
):
    from src.utils.device import dataloader_kwargs

    loader_kwargs = dataloader_kwargs(num_workers=num_workers, pin_memory=pin_memory)
    dataset = LineDataset(
        images_dir=images_dir,
        csv_file=csv_file,
        processor=processor,
        tokenizer=tokenizer,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_fn,
        **loader_kwargs,
    )


def get_line_dataloaders(batch_size=4, num_workers=None, pin_memory=True):
    """Build train/val loaders from SinOCR-handwritten and SinOCR-printed."""
    from src.utils.device import dataloader_kwargs

    loader_kwargs = dataloader_kwargs(num_workers=num_workers, pin_memory=pin_memory)

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-384")
    tokenizer = AutoTokenizer.from_pretrained("keshan/SinhalaBERTo")

    train_datasets = [
        LineDataset(
            images_dir=SINOCR_HANDWRITTEN["train"]["images_dir"],
            csv_file=SINOCR_HANDWRITTEN["train"]["csv_file"],
            processor=processor,
            tokenizer=tokenizer,
        ),
        LineDataset(
            images_dir=SINOCR_PRINTED["train"]["images_dir"],
            csv_file=SINOCR_PRINTED["train"]["csv_file"],
            processor=processor,
            tokenizer=tokenizer,
        ),
    ]
    val_datasets = [
        LineDataset(
            images_dir=SINOCR_HANDWRITTEN["test"]["images_dir"],
            csv_file=SINOCR_HANDWRITTEN["test"]["csv_file"],
            processor=processor,
            tokenizer=tokenizer,
        ),
        LineDataset(
            images_dir=SINOCR_PRINTED["test"]["images_dir"],
            csv_file=SINOCR_PRINTED["test"]["csv_file"],
            processor=processor,
            tokenizer=tokenizer,
        ),
    ]

    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        ConcatDataset(val_datasets),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        **loader_kwargs,
    )
    return train_loader, val_loader
