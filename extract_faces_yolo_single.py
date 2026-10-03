import os
import cv2
import argparse
import torch
from ultralytics import YOLO

# ---------------- CONFIG ----------------
SAMPLE_RATE = 3
MAX_FACES = 150
MIN_CONFIDENCE = 0.4
MIN_FACE_SIZE = 30
FACE_MARGIN = 0.2
FACE_SIZE = 224
# ---------------------------------------


def extract_faces(video_path, out_dir, model, device="cpu"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    os.makedirs(out_dir, exist_ok=True)

    frame_idx = 0
    count = 0

    while cap.isOpened() and count < MAX_FACES:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % SAMPLE_RATE != 0:
            continue

        h, w = frame.shape[:2]

        results = model(frame, conf=MIN_CONFIDENCE, device=device, verbose=False)

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

        crop = cv2.resize(crop, (FACE_SIZE, FACE_SIZE))
        out_path = os.path.join(out_dir, f"{count:05d}.jpg")
        cv2.imwrite(out_path, crop)

        count += 1

    cap.release()

    if count == 0:
        raise RuntimeError(f"No faces detected in '{video_path}'. Try lowering confidence or checking video quality.")

    print(f"✓ Successfully extracted {count} face frames to: {out_dir}")


if __name__ == "__main__":
    default_model = "yolov8m.pt"
    if not os.path.exists(default_model):
        default_model = "yolov8n.pt"

    parser = argparse.ArgumentParser(description="Extract aligned face crops from video using YOLO")
    parser.add_argument("--video", required=True, help="Path to input video (.mp4, .avi, etc.)")
    parser.add_argument("--out", required=True, help="Output directory to save extracted face crops")
    parser.add_argument(
        "--model",
        default=default_model,
        help=f"Path or name of YOLO model weights. Default: {default_model}"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    yolo = YOLO(args.model)
    extract_faces(args.video, args.out, yolo, device=device)
