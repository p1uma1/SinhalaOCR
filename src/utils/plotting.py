import json
import os

import matplotlib.pyplot as plt


def _epochs(history):
    return list(range(1, len(history) + 1))


def save_history(history, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return path


def plot_stage1_metrics(history, output_dir):
    """Plot Stage 1 loss and accuracy curves."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = _epochs(history["train_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="Validation", marker="o", markersize=3)
    axes[0].set_title("Stage 1 — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"], label="Validation", marker="o", markersize=3)
    axes[1].set_title("Stage 1 — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    combined_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)

    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    ax_loss.plot(epochs, history["val_loss"], label="Validation", marker="o", markersize=3)
    ax_loss.set_title("Stage 1 — Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    fig_loss.tight_layout()
    loss_path = os.path.join(output_dir, "loss_curve.png")
    fig_loss.savefig(loss_path, dpi=150)
    plt.close(fig_loss)

    fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
    ax_acc.plot(epochs, history["train_acc"], label="Train", marker="o", markersize=3)
    ax_acc.plot(epochs, history["val_acc"], label="Validation", marker="o", markersize=3)
    ax_acc.set_title("Stage 1 — Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    fig_acc.tight_layout()
    acc_path = os.path.join(output_dir, "accuracy_curve.png")
    fig_acc.savefig(acc_path, dpi=150)
    plt.close(fig_acc)

    return combined_path, loss_path, acc_path


def plot_stage2_metrics(history, output_dir):
    """Plot Stage 2 loss, WER, and CER curves."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = _epochs(history["train_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="Validation", marker="o", markersize=3)
    axes[0].set_title("Stage 2 — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["wer"], label="WER", marker="o", markersize=3, color="tab:orange")
    axes[1].plot(epochs, history["cer"], label="CER", marker="o", markersize=3, color="tab:green")
    axes[1].set_title("Stage 2 — WER / CER")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    combined_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)

    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    ax_loss.plot(epochs, history["val_loss"], label="Validation", marker="o", markersize=3)
    ax_loss.set_title("Stage 2 — Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    fig_loss.tight_layout()
    loss_path = os.path.join(output_dir, "loss_curve.png")
    fig_loss.savefig(loss_path, dpi=150)
    plt.close(fig_loss)

    fig_metrics, ax_metrics = plt.subplots(figsize=(8, 5))
    ax_metrics.plot(epochs, history["wer"], label="WER", marker="o", markersize=3, color="tab:orange")
    ax_metrics.plot(epochs, history["cer"], label="CER", marker="o", markersize=3, color="tab:green")
    ax_metrics.set_title("Stage 2 — WER / CER")
    ax_metrics.set_xlabel("Epoch")
    ax_metrics.set_ylabel("Error Rate")
    ax_metrics.legend()
    ax_metrics.grid(True, alpha=0.3)
    fig_metrics.tight_layout()
    metrics_path = os.path.join(output_dir, "wer_cer_curve.png")
    fig_metrics.savefig(metrics_path, dpi=150)
    plt.close(fig_metrics)

    return combined_path, loss_path, metrics_path
