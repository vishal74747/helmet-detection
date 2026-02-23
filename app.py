import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile, os, time, io
import urllib.request

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .title { text-align:center; color:#4A9EFF; font-size:42px;
             font-weight:800; margin-bottom:0; }
    .subtitle { text-align:center; color:#888; font-size:16px;
                margin-top:4px; margin-bottom:30px; }
    .metric-box { background:#1e2130; border-radius:12px; padding:20px;
                  text-align:center; border:1px solid #2d3150; }
    .metric-val { font-size:32px; font-weight:800; color:#4A9EFF; }
    .metric-label { font-size:13px; color:#888; margin-top:4px; }
    .alert-red { background:#3d1515; border:1px solid #ff4444;
                 border-radius:8px; padding:12px 20px;
                 color:#ff6666; font-weight:600; font-size:15px; }
    .alert-green { background:#0d2e1a; border:1px solid #22c55e;
                   border-radius:8px; padding:12px 20px;
                   color:#4ade80; font-weight:600; font-size:15px; }
</style>
""", unsafe_allow_html=True)

# ── Download & Load Model ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # If the model file isn't present, download from Hugging Face
    if not os.path.exists("best.pt"):
        url = "https://huggingface.co/justlikethat06/helmet-model/resolve/main/best.pt"
        urllib.request.urlretrieve(url, "best.pt")
    return YOLO("best.pt")

# ── Helper ────────────────────────────────────────────────────────────────────
def run_predict(model, img_array, conf):
    results  = model.predict(img_array, conf=conf, verbose=False)[0]
    annotated = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
    helmet   = sum(1 for c in results.boxes.cls if model.names[int(c)].lower()=='helmet')
    nohelmet = sum(1 for c in results.boxes.cls if model.names[int(c)].lower()=='nohelmet')
    return annotated, helmet, nohelmet

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="title">🪖 Helmet Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">YOLOv8m · Real-time Detection</p>',
            unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🪖 Helmet Detection")
    mode = st.radio("Detection Mode", ["📷 Image","🎥 Video"])
    conf = st.slider("Confidence Threshold", 0.1, 0.95, 0.4, 0.05)

# ── Load model ─────────────────────────────────────────────────────────────────
try:
    model = load_model()
    st.sidebar.success("✅ Model loaded!")
except Exception as e:
    st.error(f"❌ Model loading failed:\n{e}")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE MODE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "📷 Image":
    st.subheader("📷 Image Detection")
    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp","webp"])

    if uploaded:
        img       = Image.open(uploaded).convert("RGB")
        img_array = np.array(img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)

        with col2:
            with st.spinner("Running detection..."):
                annotated, helmet, nohelmet = run_predict(model, img_array, conf)
            st.image(annotated, caption="Detection Result", use_container_width=True)

        st.markdown("---")
        st.metric("🟢 With Helmet", helmet)
        st.metric("🔴 Without Helmet", nohelmet)
        st.metric("👥 Total Detected", helmet + nohelmet)

        if nohelmet > 0:
            st.error(f"⚠️ {nohelmet} person(s) without helmet!")
        else:
            st.success("✅ All clear!")

# ══════════════════════════════════════════════════════════════════════════════
# VIDEO MODE
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🎥 Video":
    st.subheader("🎥 Video Detection")
    uploaded = st.file_uploader("Choose a video", type=["mp4","avi","mov","mkv"])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, helmet, nohelmet = run_predict(model, frame, conf)
            stframe.image(annotated, channels="RGB", use_container_width=True)

        cap.release()
