import os
import sys
import argparse
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

# Setup module path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from model import DeepfakeDetector

# Default configurations
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "best_model.pth")
DEFAULT_YOLO_PATH = os.path.join(PROJECT_ROOT, "yolov8m.pt")
if not os.path.exists(DEFAULT_YOLO_PATH):
    DEFAULT_YOLO_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")
if not os.path.exists(DEFAULT_YOLO_PATH):
    DEFAULT_YOLO_PATH = "yolov8n.pt"

SEQUENCE_LENGTH = 30
SAMPLE_RATE = 3
MIN_CONFIDENCE = 0.4
FACE_MARGIN = 0.2
MIN_FACE_SIZE = 30
MAX_FACES = 150

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalization matching training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_faces(video_path, yolo):
    """Extract face crops from video frames using YOLO detection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    faces = []
    frame_idx = 0

    while cap.isOpened() and len(faces) < MAX_FACES:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % SAMPLE_RATE != 0:
            continue

        h, w = frame.shape[:2]
        results = yolo(frame, conf=MIN_CONFIDENCE, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            continue

        largest_box = None
        max_area = 0

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                largest_box = (x1, y1, x2, y2)
                max_area = area

        if largest_box is None:
            continue

        x1, y1, x2, y2 = largest_box
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * FACE_MARGIN), int(bh * FACE_MARGIN)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < MIN_FACE_SIZE or crop.shape[1] < MIN_FACE_SIZE:
            continue

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (224, 224))
        faces.append(crop)

    cap.release()
    return faces


def prepare_sequence(faces):
    """Uniform temporal sampling into a (1, T, C, H, W) tensor sequence."""
    if len(faces) == 0:
        raise ValueError("No faces detected in the provided video.")

    if len(faces) >= SEQUENCE_LENGTH:
        idx = np.linspace(0, len(faces) - 1, SEQUENCE_LENGTH).astype(int)
        sampled = [faces[i] for i in idx]
    else:
        sampled = list(faces)
        while len(sampled) < SEQUENCE_LENGTH:
            sampled.append(sampled[-1])

    tensors = [transform(Image.fromarray(f)) for f in sampled]
    return torch.stack(tensors).unsqueeze(0).to(DEVICE)


def predict_video(video_path, model_path=DEFAULT_MODEL_PATH, yolo_path=DEFAULT_YOLO_PATH):
    """Run full inference pipeline on an input video."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at '{model_path}'. "
            "Please ensure 'best_model.pth' is downloaded and in place."
        )

    print(f"Device: {DEVICE.upper()}")
    print(f"Loading YOLO face detector from: {yolo_path}")
    yolo = YOLO(yolo_path).to(DEVICE)

    print(f"Loading DeepfakeDetector from: {model_path}")
    model = DeepfakeDetector().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print(f"Extracting faces from: {video_path}")
    faces = extract_faces(video_path, yolo)
    print(f"Total candidate faces extracted: {len(faces)}")

    sequence = prepare_sequence(faces)

    with torch.no_grad():
        logits = model(sequence)
        prob_fake = torch.sigmoid(logits).item()

    label = "FAKE" if prob_fake > 0.5 else "REAL"
    confidence = max(prob_fake, 1 - prob_fake)

    return {
        "prediction": label,
        "fake_probability": prob_fake,
        "real_probability": 1.0 - prob_fake,
        "confidence": confidence,
        "faces_detected": len(faces)
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deepfake Video Detection using CNN-BiLSTM Pipeline"
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to input video file (e.g. sample.mp4)"
    )
    parser.add_argument(
        "--video",
        dest="video_flag",
        help="Path to input video file"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to model checkpoint (.pth). Default: {DEFAULT_MODEL_PATH}"
    )
    parser.add_argument(
        "--yolo",
        default=DEFAULT_YOLO_PATH,
        help=f"Path to YOLO face model. Default: {DEFAULT_YOLO_PATH}"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_input = args.video_flag or args.video

    if not video_input:
        print("Usage: python predict_video.py <path_to_video.mp4>")
        print("       python predict_video.py --video <path_to_video.mp4> --model best_model.pth")
        sys.exit(1)

    result = predict_video(video_input, model_path=args.model, yolo_path=args.yolo)

    print("\n" + "=" * 55)
    print("           DEEPFAKE PREDICTION RESULT")
    print("=" * 55)
    for k, v in result.items():
        if isinstance(v, float):
            print(f" {k:22}: {v:.4f} ({v*100:.2f}%)")
        else:
            print(f" {k:22}: {v}")
    print("=" * 55)
