import os
import sys
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from model import DeepfakeDetector
from dataset import VideoDataset

# ---------------- CONFIG ----------------
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "processed")
TEST_SPLIT_FILE = os.path.join(PROJECT_ROOT, "splits", "test.txt")
MODEL_PATH = os.path.join(PROJECT_ROOT, "best_model.pth")

BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------- DATASET ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

if not os.path.exists(TEST_SPLIT_FILE):
    print(f"Error: Test split '{TEST_SPLIT_FILE}' not found.")
    sys.exit(1)

test_dataset = VideoDataset(
    root=DATA_ROOT,
    split_file=TEST_SPLIT_FILE,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True if DEVICE.type == "cuda" else False
)

print(f"Test samples: {len(test_dataset)}")

# ---------------- MODEL ----------------
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model weights '{MODEL_PATH}' not found.")
    sys.exit(1)

model = DeepfakeDetector().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------- EVALUATION ----------------
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for videos, labels in tqdm(test_loader, desc="Evaluating Test Set"):
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(videos).squeeze(1)
        probs = torch.sigmoid(logits)

        preds = (probs >= 0.5).long()

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_probs)

print("\n" + "=" * 50)
print("             TEST SET EVALUATION RESULTS")
print("=" * 50)
print(f" Accuracy      : {accuracy * 100:.2f}% ({accuracy:.4f})")
print(f" Precision     : {precision * 100:.2f}% ({precision:.4f})")
print(f" Recall        : {recall * 100:.2f}% ({recall:.4f})")
print(f" F1-Score      : {f1 * 100:.2f}% ({f1:.4f})")
print(f" ROC AUC       : {roc_auc:.4f}")
print("=" * 50)
