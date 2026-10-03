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
DEFAULT_NANO = os.path.join(PROJECT_ROOT, "yolov8n.pt")
DEFAULT_MED = os.path.join(PROJECT_ROOT, "yolov8m.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQUENCE_LENGTH = 30
MIN_CONFIDENCE = 0.4
FACE_MARGIN = 0.2
MIN_FACE_SIZE = 30
FACE_SIZE = 224
THRESHOLD = 0.5

# ---------------- COMPATIBILITY HELPER ----------------
def render_image(target, img, caption=None):
    """Safely render image without deprecation warnings across Streamlit versions."""
    try:
        target.image(img, caption=caption, width="stretch")
    except TypeError:
        target.image(img, caption=caption, use_container_width=True)

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
def load_models(yolo_weight_file):
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Model checkpoint '{MODEL_PATH}' was not found. Please ensure best_model.pth is present."

    try:
        model = DeepfakeDetector().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except Exception as e:
        return None, None, f"Error loading PyTorch model: {str(e)}"

    try:
        yolo = YOLO(yolo_weight_file)
    except Exception as e:
        return None, None, f"Error loading YOLO face detector: {str(e)}"

    return model, yolo, None

# ---------------- OPTIMIZED FACE EXTRACTION ----------------
def extract_faces_optimized(video_path, yolo, progress_callback=None):
    """
    Uniformly samples frames across the entire video duration to gather
    the required sequence frames quickly without scanning hundreds of redundant frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 150

    # Target 36 candidate points evenly spread throughout the video
    num_samples = max(SEQUENCE_LENGTH, min(40, total_frames))
    target_frame_indices = set(np.linspace(0, max(0, total_frames - 1), num=num_samples, dtype=int))

    faces = []
    frame_idx = 0
    processed_count = 0
    total_targets = len(target_frame_indices)

    while cap.isOpened() and len(faces) < SEQUENCE_LENGTH + 6:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in target_frame_indices:
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count / total_targets, f"Analyzing frame {processed_count} of {total_targets}...")

            h, w = frame.shape[:2]
            results = yolo(frame, conf=MIN_CONFIDENCE, device=DEVICE, verbose=False)

            if len(results) > 0 and len(results[0].boxes) > 0:
                largest = None
                max_area = 0

                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        largest = (x1, y1, x2, y2)
                        max_area = area

                if largest is not None:
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
                    if crop.size > 0 and crop.shape[0] >= MIN_FACE_SIZE and crop.shape[1] >= MIN_FACE_SIZE:
                        crop = cv2.resize(crop, (FACE_SIZE, FACE_SIZE))
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        faces.append(crop)

        frame_idx += 1

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

def predict_video(video_path, model, yolo, progress_bar, status_text):
    def update_progress(pct, msg):
        progress_bar.progress(min(0.85, pct * 0.85))
        status_text.text(msg)

    faces = extract_faces_optimized(video_path, yolo, progress_callback=update_progress)
    if len(faces) == 0:
        return None, None

    status_text.text("Constructing 30-frame spatiotemporal tensor...")
    progress_bar.progress(0.90)

    seq_res = make_sequence(faces)
    if seq_res is None:
        return None, None

    sequence, sampled_faces = seq_res

    status_text.text("Running Bidirectional LSTM temporal analysis...")
    progress_bar.progress(0.96)

    with torch.inference_mode():
        logit = model(sequence)
        prob_fake = torch.sigmoid(logit).item()

    progress_bar.progress(1.0)
    status_text.text("Analysis complete!")

    return prob_fake, sampled_faces

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🛡️ Deepfake Guard")
    st.markdown("**CNN-BiLSTM Spatiotemporal Detection**")
    st.markdown("---")
    
    # Model speed toggle
    model_choice = st.selectbox(
        "Face Detector Model",
        options=["YOLOv8 Nano (Fast / CPU Recommended)", "YOLOv8 Medium (Heavy)"],
        index=0 if DEVICE == "cpu" else 1
    )
    selected_yolo = DEFAULT_NANO if "Nano" in model_choice else DEFAULT_MED
    if not os.path.exists(selected_yolo):
        selected_yolo = "yolov8n.pt"

    st.markdown(f"**Hardware Device:** `{DEVICE.upper()}`")
    st.markdown(f"**Target Sequence:** `{SEQUENCE_LENGTH} frames`")
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

    model, yolo, err = load_models(selected_yolo)
    if err:
        st.error(err)
        st.info("Tip: Ensure 'best_model.pth' and 'yolov8n.pt' / 'yolov8m.pt' exist in the project root directory.")
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
                run_btn = st.button("🚀 Analyze Video", type="primary")

                if run_btn:
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    prob_fake, sampled_faces = predict_video(
                        video_path, model, yolo, progress_bar, status_text
                    )

                    if prob_fake is None:
                        st.error("❌ No valid human faces detected in the video stream. Ensure the face is clearly visible and well-lit.")
                    else:
                        is_fake = prob_fake > THRESHOLD
                        label = "FAKE" if is_fake else "REAL"
                        badge_color = "#e63946" if is_fake else "#2a9d8f"
                        verdict_emoji = "🚨" if is_fake else "✅"
                        confidence = max(prob_fake, 1 - prob_fake)

                        st.markdown(
                            f"""
                            <div style="background: rgba(255,255,255,0.06); padding: 20px; border-radius: 12px; border-left: 6px solid {badge_color}; margin-top: 15px; margin-bottom: 20px;">
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
                                    render_image(face_cols[idx], f_crop, caption=f"Frame {idx+1}")

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
            render_image(st, roc_img, caption="ROC Curve (AUC = 0.9429)")
        if os.path.exists(cm_img):
            render_image(st, cm_img, caption="Confusion Matrix (Real: 180/45, Fake: 18/207)")

    with c2:
        if os.path.exists(pr_img):
            render_image(st, pr_img, caption="Precision-Recall Curve")
        if os.path.exists(prob_img):
            render_image(st, prob_img, caption="Prediction Probability Distribution")

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
       - Frames are uniformly sampled across the video sequence.
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