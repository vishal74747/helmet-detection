# 🪖 Helmet Detection using YOLOv8

A deep learning project to detect **helmet** and **no helmet** in real-time using YOLOv8 object detection model. This system can be used for **road safety monitoring** and **construction site safety enforcement**.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![mAP](https://img.shields.io/badge/mAP@50-93.7%25-brightgreen)

---

## 📌 Project Overview

This project uses a custom dataset of ~1180 images (including frames extracted from a video) to train a YOLOv8 model to detect whether a person is wearing a helmet or not.

**Use Cases:**
- 🚦 Road safety monitoring (bike riders without helmets)
- 🏗️ Construction site safety enforcement
- 📷 CCTV surveillance automation

---

## 📊 Model Results

### Validation Results
| Metric | Score |
|--------|-------|
| **mAP@50** | **93.7%** |
| **mAP@50-95** | **53.3%** |
| **Precision** | **90.7%** |
| **Recall** | **85.3%** |

### Per Class Performance
| Class | AP@50 |
|-------|-------|
| 🪖 Helmet | **95.37%** |
| ❌ No Helmet | **92.10%** |

---

## 🗂️ Dataset

| Source | Count |
|--------|-------|
| With Helmet Images | ~500 |
| Without Helmet Images | ~320 |
| Extracted from Video | ~360 |
| **Total** | **~1180** |

- Annotated using **Roboflow** (bounding box)
- Exported in **YOLO format**
- Split: **80% train / 10% val / 10% test**
- Classes: `helmet`, `nohelmet`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| YOLOv8 (Ultralytics) | Object Detection Model |
| OpenCV | Video frame extraction |
| Roboflow | Image annotation |
| Kaggle (Tesla P100 GPU) | Model training |
| Python 3.12 | Programming language |

---

## 📁 Project Structure

```
helmet-detection/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── extracted_frames/
├── runs/
│   └── helmet_detection/
│       └── weights/
│           ├── best.pt       ← Best model
│           └── last.pt       ← Last epoch model
├── helmet.yaml               ← Dataset config
└── train.py
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/helmet-detection.git
cd helmet-detection
```

### 2. Install Dependencies
```bash
pip install ultralytics opencv-python roboflow
```

### 3. Extract Frames from Video
```python
import cv2, os

cap = cv2.VideoCapture('video.mp4')
os.makedirs('extracted_frames', exist_ok=True)

count, saved = 0, 0
frame_interval = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_interval == 0:
        cv2.imwrite(f'extracted_frames/frame_{saved}.jpg', frame)
        saved += 1
    count += 1

cap.release()
print(f"Total frames saved: {saved}")
```

### 4. Train the Model
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # pretrained weights

model.train(
    data='helmet.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='helmet_detection'
)
```

### 5. Validate the Model
```python
from ultralytics import YOLO

model = YOLO('runs/helmet_detection/weights/best.pt')
metrics = model.val()

print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"Precision: {metrics.box.p:.4f}")
print(f"Recall: {metrics.box.r:.4f}")
```

### 6. Run Inference

**On Image:**
```python
model.predict(source='test.jpg', save=True, conf=0.5)
```

**On Video:**
```python
model.predict(source='video.mp4', save=True, conf=0.5)
```

**On Webcam:**
```python
model.predict(source=0, save=True, conf=0.5, show=True)
```

---

## ⚙️ helmet.yaml Config

```yaml
path: /kaggle/working/dataset
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: helmet
  1: nohelmet
```

---

## 🖥️ Training Environment

| Hardware | Details |
|----------|---------|
| GPU | Tesla P100 16GB |
| Platform | Kaggle Notebooks |
| Framework | PyTorch + Ultralytics |
| Training Time | ~15-20 mins |

---

## 📈 Training Highlights

- ✅ No overfitting — train and test results are close
- ✅ Strong per-class performance (95.37% helmet, 92.10% no helmet)
- ✅ Model size: **52MB** (best.pt)
- ✅ Inference speed: **~11.5ms per image**

---

## 🔮 Future Improvements

- [ ] Increase dataset size to 3000+ images
- [ ] Deploy as a web app using Streamlit or Gradio
- [ ] Real-time CCTV integration
- [ ] Mobile deployment using TensorFlow Lite
- [ ] Alert system when no helmet is detected

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com) — for annotation
- [Kaggle](https://kaggle.com) — for free GPU

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Vishal**  
[![Kaggle](https://img.shields.io/badge/Kaggle-vishal747-blue)](https://kaggle.com/vishal747)
