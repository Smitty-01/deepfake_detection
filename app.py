import os
import sys
import tempfile
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# ---------------- PATH SETUP ----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from model import DeepfakeDetector

# ---------------- CONFIG ----------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "best_model.pth")
YOLO_MODEL = os.path.join(PROJECT_ROOT, "yolov8m.pt")
if not os.path.exists(YOLO_MODEL):
    YOLO_MODEL = os.path.join(PROJECT_ROOT, "yolov8n.pt")
if not os.path.exists(YOLO_MODEL):
    YOLO_MODEL = "yolov8n.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQUENCE_LENGTH = 30
SAMPLE_RATE = 3
MIN_CONFIDENCE = 0.4
FACE_MARGIN = 0.2
MIN_FACE_SIZE = 30
MAX_FACES = 150
FACE_SIZE = 224
THRESHOLD = 0.5

# ---------------- PAGE CONFIG & STYLING ----------------
st.set_page_config(
    page_title="Deepfake Video Detector | CNN-BiLSTM",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS stylesheet
css_path = os.path.join(PROJECT_ROOT, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b132b, #1c2541, #3a506b);
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((FACE_SIZE, FACE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- CACHED MODEL LOADER ----------------
@st.cache_resource(show_spinner=False)
def load_models():
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Model checkpoint '{MODEL_PATH}' was not found. Please ensure best_model.pth is present."

    try:
        model = DeepfakeDetector().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except Exception as e:
        return None, None, f"Error loading PyTorch model: {str(e)}"

    try:
        yolo = YOLO(YOLO_MODEL)
    except Exception as e:
        return None, None, f"Error loading YOLO face detector: {str(e)}"

    return model, yolo, None

# ---------------- FACE EXTRACTION ----------------
def extract_faces(video_path, yolo):
    faces = []
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened() and len(faces) < MAX_FACES:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % SAMPLE_RATE != 0:
            continue

        h, w = frame.shape[:2]
        results = yolo(frame, conf=MIN_CONFIDENCE, device=DEVICE, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            continue

        largest = None
        max_area = 0

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                largest = (x1, y1, x2, y2)
                max_area = area

        if largest is None:
            continue

        x1, y1, x2, y2 = largest
        bw = x2 - x1
        bh = y2 - y1
        mx = int(bw * FACE_MARGIN)
        my = int(bh * FACE_MARGIN)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < MIN_FACE_SIZE or crop.shape[1] < MIN_FACE_SIZE:
            continue

        crop = cv2.resize(crop, (FACE_SIZE, FACE_SIZE))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        faces.append(crop)

    cap.release()
    return faces

def make_sequence(faces):
    if len(faces) == 0:
        return None

    if len(faces) >= SEQUENCE_LENGTH:
        idx = np.linspace(0, len(faces) - 1, SEQUENCE_LENGTH).astype(int)
        sampled = [faces[i] for i in idx]
    else:
        sampled = list(faces)
        while len(sampled) < SEQUENCE_LENGTH:
            sampled.append(sampled[-1])

    tensor = torch.stack([
        transform(Image.fromarray(face)) for face in sampled
    ]).unsqueeze(0).to(DEVICE, non_blocking=True)

    return tensor, sampled

def predict_video(video_path, model, yolo):
    faces = extract_faces(video_path, yolo)
    if len(faces) == 0:
        return None, None

    seq_res = make_sequence(faces)
    if seq_res is None:
        return None, None

    sequence, sampled_faces = seq_res

    with torch.inference_mode():
        logit = model(sequence)
        prob_fake = torch.sigmoid(logit).item()

    return prob_fake, sampled_faces

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🛡️ Deepfake Guard")
    st.markdown("**CNN-BiLSTM Spatiotemporal Detection**")
    st.markdown("---")
    st.markdown(f"**Hardware Device:** `{DEVICE.upper()}`")
    st.markdown(f"**Target Sequence:** `{SEQUENCE_LENGTH} frames`")
    st.markdown(f"**Sampling Rate:** `Every {SAMPLE_RATE} frames`")
    st.markdown(f"**Classification Threshold:** `{THRESHOLD}`")
    st.markdown("---")
    st.markdown("### 📚 Research Details")
    st.markdown("""
    **Project:** High-Fidelity Deepfake Detection
    **Event:** NIRMAAN 2026, SVKM IT
    **Authors:**
    - Ashmit Kinariwala
    - Aaryan Kamdar
    - Atharv Kulkarni
    - Darshan Purohit
    """)
    st.markdown("---")
    st.caption("ResNet50 + Bidirectional LSTM Video Classifier")

# ---------------- MAIN APPLICATION ----------------
tab_infer, tab_metrics, tab_arch = st.tabs([
    "🔍 Live Detection",
    "📊 Evaluation Metrics",
    "🧠 Model Architecture"
])

# ----- TAB 1: INFERENCE -----
with tab_infer:
    st.title("🛡️ High-Fidelity Deepfake Video Detection")
    st.markdown(
        "Upload a video file to evaluate facial temporal continuity and identify deepfake manipulation."
    )

    model, yolo, err = load_models()
    if err:
        st.error(err)
        st.info("Tip: Ensure 'best_model.pth' and 'yolov8m.pt' / 'yolov8n.pt' exist in the project root directory.")
    else:
        uploaded_video = st.file_uploader(
            "Select a video (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"]
        )

        if uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                video_path = tmp.name

            col_vid, col_pred = st.columns([1, 1])

            with col_vid:
                st.subheader("📹 Input Video")
                st.video(uploaded_video)

            with col_pred:
                st.subheader("⚡ Analysis & Results")
                run_btn = st.button("🚀 Analyze Video", use_container_width=True, type="primary")

                if run_btn:
                    with st.spinner("Extracting facial frames and evaluating temporal patterns..."):
                        prob_fake, sampled_faces = predict_video(video_path, model, yolo)

                    if prob_fake is None:
                        st.error("❌ No valid human faces detected in the video stream. Ensure the face is clearly visible.")
                    else:
                        is_fake = prob_fake > THRESHOLD
                        label = "FAKE" if is_fake else "REAL"
                        badge_color = "#e63946" if is_fake else "#2a9d8f"
                        verdict_emoji = "🚨" if is_fake else "✅"
                        confidence = max(prob_fake, 1 - prob_fake)

                        st.markdown(
                            f"""
                            <div style="background: rgba(255,255,255,0.06); padding: 20px; border-radius: 12px; border-left: 6px solid {badge_color}; margin-bottom: 20px;">
                                <h2 style="color: {badge_color}; margin: 0;">{verdict_emoji} VERDICT: {label}</h2>
                                <p style="margin: 5px 0 0 0; color: #e0e0e0;">
                                    {'The video exhibits temporal & spatial artifacts characteristic of deepfake synthesis.' if is_fake else 'No significant facial manipulation artifacts detected. The video appears authentic.'}
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("Manipulated Probability", f"{prob_fake * 100:.2f}%")
                        with m2:
                            st.metric("Authentic Probability", f"{(1 - prob_fake) * 100:.2f}%")

                        st.markdown("**Confidence Level:**")
                        st.progress(float(confidence))

                        if 0.40 <= prob_fake <= 0.60:
                            st.warning("⚠️ High uncertainty detected. The prediction is close to the decision boundary.")

                        if sampled_faces and len(sampled_faces) > 0:
                            with st.expander(f"👁️ Sampled Face Crops ({len(sampled_faces)} frames analyzed)"):
                                face_cols = st.columns(min(6, len(sampled_faces)))
                                for idx, f_crop in enumerate(sampled_faces[:6]):
                                    face_cols[idx].image(f_crop, caption=f"Frame {idx+1}", use_container_width=True)

# ----- TAB 2: EVALUATION METRICS -----
with tab_metrics:
    st.header("📊 Benchmark Evaluation & Test Results")
    st.markdown(
        "Performance evaluation on the independent test dataset (450 balanced video samples from FaceForensics++, DFDC, and Celeb-DF)."
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Accuracy", "86.00%", "Test Split")
    k2.metric("Precision", "82.14%", "Fake Detection")
    k3.metric("Recall", "92.00%", "High Sensitivity")
    k4.metric("F1-Score", "86.79%", "Harmonic Mean")
    k5.metric("ROC AUC", "0.9429", "Separability")

    st.markdown("---")
    st.subheader("📈 Performance Curves & Diagnostic Distributions")

    c1, c2 = st.columns(2)
    roc_img = os.path.join(PROJECT_ROOT, "assets", "roc_curve.png")
    pr_img = os.path.join(PROJECT_ROOT, "assets", "precision_recall_curve.png")
    cm_img = os.path.join(PROJECT_ROOT, "assets", "confusion_matrix.png")
    prob_img = os.path.join(PROJECT_ROOT, "assets", "probability_distribution.png")

    with c1:
        if os.path.exists(roc_img):
            st.image(roc_img, caption="ROC Curve (AUC = 0.9429)", use_container_width=True)
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Confusion Matrix (Real: 180/45, Fake: 18/207)", use_container_width=True)

    with c2:
        if os.path.exists(pr_img):
            st.image(pr_img, caption="Precision-Recall Curve", use_container_width=True)
        if os.path.exists(prob_img):
            st.image(prob_img, caption="Prediction Probability Distribution", use_container_width=True)

    st.markdown("---")
    st.subheader("🔬 Ablation Study Comparison")
    st.markdown("""
    | Architecture Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
    | :--- | :---: | :---: | :---: | :---: |
    | **CNN-Only (ResNet50)** | 63.00% | 64.39% | 58.67% | 61.40% |
    | **CNN + Unidirectional LSTM** | 81.00% | 83.00% | 81.00% | 81.00% |
    | **CNN + Bidirectional LSTM (Ours)** | **86.00%** | **82.14%** | **92.00%** | **86.79%** |
    """)

# ----- TAB 3: ARCHITECTURE -----
with tab_arch:
    st.header("🧠 End-to-End System Pipeline")
    st.markdown("""
    The pipeline combines high-capacity 2D spatial feature extraction with bidirectional recurrent modeling to capture temporal artifacts:
    
    1. **Face Detection & Tracking (YOLOv8):**
       - Frames are uniformly sampled at fixed intervals (`rate=3`).
       - YOLO detects the largest face candidate, padded by a 20% margin to capture boundary blending artifacts.
       - Faces are resized to `224x224` and normalized using standard ImageNet parameters.
    
    2. **Spatial Feature Extraction (ResNet50 Backbone):**
       - Pre-trained on ImageNet.
       - Layers 1–3 are frozen to retain general low-level vision filters.
       - Layer 4 is fine-tuned to capture domain-specific manipulation cues (warping, lighting discordance).
       - Fully connected layer replaced with Identity to yield 2048-dimensional feature representations.
    
    3. **Temporal Dynamics Modeling (Bidirectional LSTM):**
       - Takes the sequence tensor of shape `(Batch, 30, 2048)`.
       - Dual-directional LSTM (hidden dimension 256 per direction $\\rightarrow$ total 512) processes frames forward and backward to detect unnatural blinks, jitter, and frame discontinuities.
    
    4. **Classification Head:**
       - Temporal sequence mean pooling $\\rightarrow$ Linear(512, 256) $\\rightarrow$ ReLU $\\rightarrow$ Dropout(0.5) $\\rightarrow$ Linear(256, 1) $\\rightarrow$ Sigmoid activation.
    """)