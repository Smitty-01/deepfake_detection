import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from model import DeepfakeDetector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "best_model.pth")
SEQUENCE_LENGTH = 30

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_faces(folder):
    """Load ordered face crops from an image directory."""
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder '{folder}' does not exist.")

    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    images = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(valid_exts)
    ])

    if len(images) == 0:
        raise ValueError(f"No face images found in '{folder}'. Supported formats: {valid_exts}")

    faces = []
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        faces.append(transform(img))

    return faces


def make_sequence(faces):
    """Uniform temporal sampling into a (1, T, C, H, W) tensor sequence."""
    if len(faces) >= SEQUENCE_LENGTH:
        idx = np.linspace(0, len(faces) - 1, SEQUENCE_LENGTH).astype(int)
        sampled = [faces[i] for i in idx]
    else:
        sampled = list(faces)
        while len(sampled) < SEQUENCE_LENGTH:
            sampled.append(sampled[-1])

    return torch.stack(sampled).unsqueeze(0).to(DEVICE)


def predict(folder, model_path=DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at '{model_path}'.")

    faces = load_faces(folder)
    sequence = make_sequence(faces)

    model = DeepfakeDetector().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        logit = model(sequence)
        prob_fake = torch.sigmoid(logit).item()

    is_fake = prob_fake > 0.5
    confidence = max(prob_fake, 1.0 - prob_fake)

    return {
        "prediction": "FAKE" if is_fake else "REAL",
        "fake_probability": prob_fake,
        "real_probability": 1.0 - prob_fake,
        "confidence": confidence,
        "frames_loaded": len(faces)
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict deepfake probability directly from extracted face crops folder"
    )
    parser.add_argument("folder", nargs="?", help="Path to folder containing ordered face frames")
    parser.add_argument("--folder", dest="folder_flag", help="Path to faces folder")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to best_model.pth")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_folder = args.folder_flag or args.folder

    if not target_folder:
        print("Usage: python predict_from_faces.py <path_to_faces_folder>")
        print("       python predict_from_faces.py --folder <path_to_faces_folder> --model best_model.pth")
        sys.exit(1)

    result = predict(target_folder, model_path=args.model)

    print("\n" + "=" * 55)
    print("           FACE SEQUENCE PREDICTION RESULT")
    print("=" * 55)
    for k, v in result.items():
        if isinstance(v, float):
            print(f" {k:22}: {v:.4f} ({v*100:.2f}%)")
        else:
            print(f" {k:22}: {v}")
    print("=" * 55)
