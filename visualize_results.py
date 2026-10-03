import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from dataset import VideoDataset
from model import DeepfakeDetector

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(PROJECT_ROOT, "best_model.pth")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "processed")
SPLIT_FILE = os.path.join(PROJECT_ROOT, "splits", "test.txt")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

BATCH_SIZE = 4
SEQUENCE_LENGTH = 30
# ---------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

if not os.path.exists(SPLIT_FILE):
    print(f"Error: Split file '{SPLIT_FILE}' not found.")
    sys.exit(1)

test_dataset = VideoDataset(
    root=DATA_ROOT,
    split_file=SPLIT_FILE,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

print(f"Loaded {len(test_dataset)} test samples. Device: {DEVICE}")

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model checkpoint '{MODEL_PATH}' not found.")
    sys.exit(1)

model = DeepfakeDetector().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

y_true = []
y_prob = []

with torch.no_grad():
    for videos, labels in tqdm(test_loader, desc="Evaluating Test Set"):
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(videos).squeeze(1)
        probs = torch.sigmoid(outputs)

        y_true.extend(labels.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_prob = np.array(y_prob)
y_pred = (y_prob >= 0.5).astype(int)

# Set clean aesthetic style for publication graphs
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5), dpi=300)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            annot_kws={"size": 14, "weight": "bold"})
plt.xlabel("Predicted Label", fontweight="bold")
plt.ylabel("Actual Label", fontweight="bold")
plt.title("Confusion Matrix (Test Set)", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
cm_out = os.path.join(ASSETS_DIR, "confusion_matrix.png")
plt.savefig(cm_out, dpi=300)
plt.close()

# 2. Classification Report
print("\n" + "=" * 55)
print("             TEST SET CLASSIFICATION REPORT")
print("=" * 55)
print(classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4))
print("=" * 55)

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5), dpi=300)
plt.plot(fpr, tpr, color="#1f77b4", lw=2.2, label=f"Proposed BiLSTM (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="#e74c3c", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
plt.xlabel("False Positive Rate", fontweight="bold")
plt.ylabel("True Positive Rate", fontweight="bold")
plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()
roc_out = os.path.join(ASSETS_DIR, "roc_curve.png")
plt.savefig(roc_out, dpi=300)
plt.close()

# 4. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_prob)

plt.figure(figsize=(6, 5), dpi=300)
plt.plot(recall, precision, color="#2ca02c", lw=2.2, label="Precision-Recall")
plt.xlabel("Recall", fontweight="bold")
plt.ylabel("Precision", fontweight="bold")
plt.title("Precision-Recall Curve", fontsize=13, fontweight="bold", pad=12)
plt.legend(loc="lower left", frameon=True)
plt.tight_layout()
pr_out = os.path.join(ASSETS_DIR, "precision_recall_curve.png")
plt.savefig(pr_out, dpi=300)
plt.close()

# 5. Probability Distribution
plt.figure(figsize=(6, 5), dpi=300)
plt.hist(y_prob[y_true == 0], bins=30, alpha=0.65, color="#1f77b4", label="Real (Ground Truth)", edgecolor="black")
plt.hist(y_prob[y_true == 1], bins=30, alpha=0.65, color="#ff7f0e", label="Fake (Ground Truth)", edgecolor="black")
plt.xlabel("Predicted Fake Probability", fontweight="bold")
plt.ylabel("Sample Count", fontweight="bold")
plt.title("Prediction Probability Distribution", fontsize=13, fontweight="bold", pad=12)
plt.legend(frameon=True)
plt.tight_layout()
prob_out = os.path.join(ASSETS_DIR, "probability_distribution.png")
plt.savefig(prob_out, dpi=300)
plt.close()

print(f"\n✓ Generated high-resolution diagnostic plots in: {ASSETS_DIR}")
