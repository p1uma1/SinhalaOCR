import os

import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer
from jiwer import wer, cer

from src.dataset.line_dataset import get_line_dataloaders
from src.models.trocr_model import SinhalaTrOCR
from src.utils.device import configure_gpu
from src.utils.plotting import plot_stage2_metrics, save_history


def decode_batch(predictions, labels, tokenizer):
    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = labels.clone()
    labels[labels == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pairs = []
    for p, l in zip(pred_str, label_str):
        if l.strip():
            pairs.append((p.strip(), l.strip()))
    return pairs


def compute_metrics_from_pairs(pairs):
    if not pairs:
        return 0.0, 0.0
    pred_str = [p for p, _ in pairs]
    label_str = [l for _, l in pairs]
    return wer(label_str, pred_str), cer(label_str, pred_str)


def train_stage2():
    # Tuned for RTX 5090 — balances VRAM safety and throughput
    batch_size = 12
    grad_accum_steps = 3  # effective batch size 36
    gen_chunk_size = 6
    num_epochs = 25
    learning_rate = 5e-5
    max_metric_samples = 512  # WER/CER on subset — full val loss still computed
    output_dir = "outputs/stage2"

    device, use_amp = configure_gpu()

    print("Loading datasets (SinOCR-handwritten + SinOCR-printed)...")
    train_loader, val_loader = get_line_dataloaders(batch_size=batch_size)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    print(f"WER/CER computed on up to {max_metric_samples} validation samples per epoch (greedy decode)")

    print("Initializing Stage 2 Model...")
    encoder_path = "outputs/stage1/pretrained_encoder.pth"
    if not os.path.exists(encoder_path):
        print("Warning: Stage 1 encoder weights not found. Using default TrOCR encoder.")
        encoder_path = None

    model = SinhalaTrOCR(encoder_pretrained_path=encoder_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained("keshan/SinhalaBERTo")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    os.makedirs(output_dir, exist_ok=True)
    best_loss = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
        "wer": [],
        "cer": [],
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc="Training"), start=1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast(device.type):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / grad_accum_steps
                loss.backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * grad_accum_steps

        train_loss = running_loss / len(train_loader)
        print(f"Train Loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        metric_pairs = []
        metric_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                if use_amp:
                    with torch.amp.autocast(device.type):
                        outputs = model(pixel_values=pixel_values, labels=labels)
                else:
                    outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

                if metric_samples < max_metric_samples:
                    remaining = max_metric_samples - metric_samples
                    pixels = pixel_values[:remaining]
                    lbls = labels[:remaining]

                    for i in range(0, pixels.size(0), gen_chunk_size):
                        pixel_chunk = pixels[i : i + gen_chunk_size]
                        label_chunk = lbls[i : i + gen_chunk_size]
                        generated_ids = model.generate(
                            pixel_chunk,
                            num_beams=1,
                            max_length=128,
                        )
                        metric_pairs.extend(
                            decode_batch(generated_ids.cpu(), label_chunk.cpu(), tokenizer)
                        )
                    metric_samples += pixels.size(0)

        val_loss = val_loss / len(val_loader)
        wer_score, cer_score = compute_metrics_from_pairs(metric_pairs)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["wer"].append(wer_score)
        history["cer"].append(cer_score)

        print(f"Val Loss: {val_loss:.4f}, WER: {wer_score:.4f}, CER: {cer_score:.4f}")
        print(f"Metrics based on {len(metric_pairs)} validation samples")

        if val_loss < best_loss:
            best_loss = val_loss
            print("Saving best model...")
            torch.save(model.state_dict(), os.path.join(output_dir, "best_trocr.pth"))

        save_history(history, output_dir, "training_history.json")
        combined, loss_plot, metrics_plot = plot_stage2_metrics(history, output_dir)
        print(f"Saved graphs: {combined}, {loss_plot}, {metrics_plot}")

        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    train_stage2()
