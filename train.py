import argparse
import collections
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from utils.config import CLASS_NAMES, DEFAULT_MODEL_PATH, DEFAULT_NUM_WORKERS, DEFAULT_SEED, FER2013_EMOTION_TO_CLASS
from utils.data import build_datasets, build_transforms
from utils.modeling import build_model, save_checkpoint
from utils.visualization import plot_confusion_matrix


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(predictions: list[int], targets: list[int]) -> dict[str, float]:
    accuracy = float(np.mean(np.array(predictions) == np.array(targets))) if targets else 0.0
    precision = float(precision_score(targets, predictions, average="macro", zero_division=0)) if targets else 0.0
    recall = float(recall_score(targets, predictions, average="macro", zero_division=0)) if targets else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall}


def _extract_labels(dataset) -> list[int]:
    if isinstance(dataset, Subset):
        base_labels = _extract_labels(dataset.dataset)
        return [base_labels[index] for index in dataset.indices]

    if hasattr(dataset, "targets"):
        return [int(target) for target in dataset.targets]

    if hasattr(dataset, "dataframe"):
        return [int(FER2013_EMOTION_TO_CLASS[int(emotion)]) for emotion in dataset.dataframe["emotion"].tolist()]

    if hasattr(dataset, "samples"):
        return [int(sample[1]) for sample in dataset.samples]

    raise TypeError(f"Unsupported dataset type for label extraction: {type(dataset)!r}")


def _compute_class_weights(dataset, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = _extract_labels(dataset)
    counts = collections.Counter(labels)
    total = sum(counts.values())
    weights = [total / max(1, counts.get(class_index, 0)) for class_index in range(num_classes)]
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return weight_tensor / weight_tensor.mean().clamp(min=1e-6)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    all_predictions = []
    all_targets = []
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_predictions.extend(outputs.argmax(dim=1).detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    metrics = compute_metrics(all_predictions, all_targets)
    metrics["loss"] = running_loss / max(1, len(loader.dataset))
    return metrics, all_targets, all_predictions


def plot_history(history: dict[str, list[float]], output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history["train_accuracy"], label="Train")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history["train_loss"], label="Train")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train student engagement detection model")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--model-name", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "resnet50"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(DEFAULT_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, class_names = build_datasets(
        data_dir=args.data_dir,
        image_size=args.image_size,
        val_split=args.val_split,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=DEFAULT_NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=DEFAULT_NUM_WORKERS, pin_memory=torch.cuda.is_available())

    model = build_model(
        model_name=args.model_name,
        num_classes=len(class_names),
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    class_weights = _compute_class_weights(train_dataset, len(class_names), device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    best_path = output_dir / "best_model.pt"
    best_targets = []
    best_predictions = []
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics, val_targets, val_predictions = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["accuracy"])

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
        )

        improved = val_metrics["accuracy"] > best_val_accuracy or (
            val_metrics["accuracy"] == best_val_accuracy and val_metrics["loss"] < best_val_loss
        )
        if improved:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_targets = val_targets
            best_predictions = val_predictions
            epochs_without_improvement = 0
            save_checkpoint(
                path=best_path,
                model=model,
                class_names=class_names,
                image_size=args.image_size,
                model_name=args.model_name,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    plot_history(history, output_dir / "training_curves.png")

    final_report = classification_report(best_targets, best_predictions, target_names=class_names, zero_division=0)
    conf_matrix = confusion_matrix(best_targets, best_predictions)
    plot_confusion_matrix(conf_matrix, class_names, output_dir / "confusion_matrix.png")

    with open(output_dir / "evaluation.txt", "w", encoding="utf-8") as file_handle:
        file_handle.write(final_report)
        file_handle.write("\nConfusion Matrix:\n")
        file_handle.write(np.array2string(conf_matrix))
        file_handle.write(f"\nBest Validation Accuracy: {best_val_accuracy:.4f}\n")
        file_handle.write(f"Best Validation Loss: {best_val_loss:.4f}\n")

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as file_handle:
        json.dump(class_names, file_handle, indent=2)

    print(f"Best model saved to {best_path}")
    print(f"Evaluation saved to {output_dir / 'evaluation.txt'}")


if __name__ == "__main__":
    main()
