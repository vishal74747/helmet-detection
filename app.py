import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile, os, time, io

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
    .stButton>button { background:#4A9EFF; color:white; border:none;
                       border-radius:8px; font-weight:600;
                       padding:10px 24px; font-size:15px; width:100%; }
    .stButton>button:hover { background:#2d7dd2; }
    div[data-testid="metric-container"] {
        background:#1e2130; border-radius:10px;
        padding:15px; border:1px solid #2d3150;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO('best.pt')

# ── Helper ────────────────────────────────────────────────────────────────────
def run_predict(model, img_array, conf):
    results  = model.predict(img_array, conf=conf, verbose=False)[0]
    annotated = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
    helmet   = sum(1 for c in results.boxes.cls if model.names[int(c)].lower()=='helmet')
    nohelmet = sum(1 for c in results.boxes.cls if model.names[int(c)].lower()=='nohelmet')
    return annotated, helmet, nohelmet

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="title">🪖 Helmet Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">YOLOv8m &nbsp;·&nbsp; 96% mAP@0.5 &nbsp;·&nbsp; Real-time Detection</p>',
            unsafe_allow_html=True)

# ── Stats Banner ──────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
for col,(val,label) in zip([c1,c2,c3,c4],[
    ("96.0%","mAP@0.5"),("90.8%","Precision"),
    ("93.1%","Recall"),("10.5ms","Inference Speed")]):
    with col:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🪖 Helmet Detection")
    st.markdown("---")
    mode = st.radio("Detection Mode", ["📷 Image","🎥 Video","📹 Webcam"])
    conf = st.slider("Confidence Threshold", 0.1, 0.95, 0.4, 0.05)
    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("- Model: YOLOv8m")
    st.markdown("- Classes: helmet / nohelmet")
    st.markdown("- Input: 640×640 px")
    st.markdown("- Parameters: 25.8M")
    st.markdown("- Training: 150 epochs")
    st.markdown("---")
    st.markdown("**Project:** Internship — Computer Vision")

# ── Load model ────────────────────────────────────────────────────────────────
try:
    model = load_model()
    st.sidebar.success("✅ Model loaded!")
except Exception as e:
    st.error(f"❌ Could not load best.pt — make sure it's in the same folder as app.py\n\n{e}")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE MODE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "📷 Image":
    st.subheader("📷 Image Detection")
    st.markdown("Upload any image to detect helmets instantly.")

    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp","webp"])

    if uploaded:
        img       = Image.open(uploaded).convert("RGB")
        img_array = np.array(img)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📥 Original Image**")
            st.image(img, use_container_width=True)

        with col2:
            st.markdown("**🔍 Detection Result**")
            with st.spinner("Running detection..."):
                t0 = time.time()
                annotated, helmet, nohelmet = run_predict(model, img_array, conf)
                elapsed = (time.time() - t0) * 1000
            st.image(annotated, use_container_width=True)

        st.markdown("---")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("🟢 With Helmet",    helmet)
        m2.metric("🔴 Without Helmet", nohelmet)
        m3.metric("👥 Total Detected", helmet + nohelmet)
        m4.metric("⚡ Inference Time", f"{elapsed:.0f} ms")

        if nohelmet > 0:
            st.markdown(
                f'<div class="alert-red">⚠️ ALERT: {nohelmet} person(s) detected WITHOUT a helmet!</div>',
                unsafe_allow_html=True)
        elif helmet > 0:
            st.markdown(
                f'<div class="alert-green">✅ All {helmet} detected person(s) are wearing helmets.</div>',
                unsafe_allow_html=True)
        else:
            st.info("ℹ️ No persons detected. Try lowering the confidence threshold.")

        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="JPEG")
        st.download_button("⬇️ Download Result Image",
                           buf.getvalue(), "helmet_result.jpg", "image/jpeg")

# ══════════════════════════════════════════════════════════════════════════════
# VIDEO MODE
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🎥 Video":
    st.subheader("🎥 Video Detection")
    st.markdown("Upload a video to process frame by frame.")

    uploaded   = st.file_uploader("Choose a video", type=["mp4","avi","mov","mkv"])
    max_frames = st.slider("Max frames to process", 100, 1000, 300, 50)

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())
        tfile.close()

        cap   = cv2.VideoCapture(tfile.name)
        fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.info(f"📹 {W}×{H} | {fps} FPS | {total} frames total | Processing: {min(max_frames,total)} frames")

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))

        progress       = st.progress(0, text="Starting...")
        preview        = st.empty()
        frame_num      = total_h = total_nh = 0

        G,R,W_,Y,B = (0,200,0),(0,0,220),(255,255,255),(0,215,255),(255,140,0)

        while frame_num < min(max_frames, total):
            ret, frame = cap.read()
            if not ret: break

            results   = model.predict(frame, conf=conf, verbose=False)[0]
            annotated = frame.copy()
            hc = nc = 0

            for box in results.boxes:
                cid        = int(box.cls[0]); cf = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                lbl        = model.names[cid]; is_h = lbl.lower()=='helmet'
                col        = G if is_h else R
                hc        += is_h; nc += not is_h
                cv2.rectangle(annotated,(x1,y1),(x2,y2),col,2)
                txt        = f"{'HELMET' if is_h else 'NO HELMET'} {cf:.0%}"
                tw,th      = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)[0]
                cv2.rectangle(annotated,(x1,y1-th-10),(x1+tw+6,y1),col,-1)
                cv2.putText(annotated,txt,(x1+3,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.55,W_,2)

            cv2.rectangle(annotated,(0,0),(W,80),(15,15,15),-1)
            cv2.rectangle(annotated,(0,0),(W,80),B,2)
            cv2.putText(annotated,'HELMET DETECTION SYSTEM',(12,30),cv2.FONT_HERSHEY_DUPLEX,0.85,Y,2)
            cv2.putText(annotated,f'Helmet:{hc}',(12,62),cv2.FONT_HERSHEY_SIMPLEX,0.6,G,2)
            cv2.putText(annotated,f'No Helmet:{nc}',(190,62),cv2.FONT_HERSHEY_SIMPLEX,0.6,R,2)
            cv2.putText(annotated,f'Frame:{frame_num+1}/{min(max_frames,total)}',(W-210,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,W_,1)
            if nc > 0:
                cv2.rectangle(annotated,(0,H-45),(W,H),(0,0,160),-1)
                cv2.putText(annotated,f'ALERT: {nc} person(s) without helmet!',
                            (12,H-12),cv2.FONT_HERSHEY_DUPLEX,0.7,W_,2)

            writer.write(annotated)
            total_h  += hc
            total_nh += nc
            frame_num += 1

            if frame_num % 10 == 0:
                preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                              caption=f"Frame {frame_num}", use_container_width=True)
                progress.progress(frame_num / min(max_frames,total),
                                  text=f"Processing {frame_num}/{min(max_frames,total)} frames...")

        cap.release(); writer.release()
        progress.empty(); preview.empty()

        st.success(f"✅ Processed {frame_num} frames!")
        m1,m2,m3 = st.columns(3)
        m1.metric("🟢 Helmet Detections",    total_h)
        m2.metric("🔴 Violation Detections", total_nh)
        m3.metric("🎞️ Frames Processed",    frame_num)

        with open(out_path,'rb') as f:
            st.download_button("⬇️ Download Annotated Video",
                               f.read(), "helmet_output.mp4", "video/mp4")

        os.unlink(tfile.name)
        os.unlink(out_path)

# ══════════════════════════════════════════════════════════════════════════════
# WEBCAM MODE
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📹 Webcam":
    st.subheader("📹 Live Webcam Detection")
    st.markdown("Tick the checkbox below to start your webcam.")
    st.warning("⚠️ Make sure to allow browser/Python access to your webcam.")

    run = st.checkbox("▶️ Start Webcam Detection")

    stframe     = st.empty()
    col1,col2,col3 = st.columns(3)
    helmet_ph   = col1.empty()
    nohelmet_ph = col2.empty()
    fps_ph      = col3.empty()
    alert_ph    = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        prev_time = time.time()

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Cannot read from webcam. Check connection.")
                break

            annotated, helmet, nohelmet = run_predict(model, frame, conf)
            curr_time = time.time()
            fps_val   = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            stframe.image(annotated, channels="RGB", use_container_width=True)
            helmet_ph.metric("🟢 With Helmet",    helmet)
            nohelmet_ph.metric("🔴 Without Helmet", nohelmet)
            fps_ph.metric("⚡ FPS", f"{fps_val:.1f}")

            if nohelmet > 0:
                alert_ph.markdown(
                    f'<div class="alert-red">⚠️ ALERT: {nohelmet} person(s) without helmet!</div>',
                    unsafe_allow_html=True)
            else:
                alert_ph.markdown(
                    '<div class="alert-green">✅ All clear — helmets detected.</div>',
                    unsafe_allow_html=True)

            run = st.session_state.get("▶️ Start Webcam Detection", False)

        cap.release()
        st.info("📹 Webcam stopped.")
```

---

## Your folder should look like this:
```
C:\helmet_app\
    ├── app.py          ← paste code above
    ├── best.pt         ← your downloaded model
    └── requirements.txt
```

## requirements.txt:
```
streamlit
ultralytics
opencv-python
pillow
numpy