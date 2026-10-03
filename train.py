import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from model import DeepfakeDetector
from loaders import get_loaders


def train(epochs=12, batch_size=4, lr=3e-4, num_workers=2, save_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 55)
    print("      DEEPFAKE DETECTOR TRAINING PIPELINE")
    print("=" * 55)
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU Name      : {torch.cuda.get_device_name(0)}")
    print(f"Epochs        : {epochs}")
    print(f"Batch Size    : {batch_size}")
    print(f"Learning Rate : {lr}")
    print("=" * 55)

    # Load datasets
    train_loader, val_loader = get_loaders(batch_size=batch_size, num_workers=num_workers)

    # Initialize model
    model = DeepfakeDetector().to(device)

    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # -------- TRAIN LOOP --------
        model.train()
        train_loss = 0.0

        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]"):
            videos = videos.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()

            logits = model(videos)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------- VALIDATION LOOP --------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]"):
                videos = videos.to(device)
                labels = labels.to(device).unsqueeze(1)

                logits = model(videos)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_acc * 100:.2f}%"
        )

        # -------- SAVE BEST MODEL --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ⭐ Saved new best checkpoint to '{save_path}' (Val Loss: {val_loss:.4f})")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-BiLSTM Deepfake Detector")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader num_workers")
    parser.add_argument("--save-path", type=str, default="best_model.pth", help="Path to save best weights")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.workers,
        save_path=args.save_path
    )
