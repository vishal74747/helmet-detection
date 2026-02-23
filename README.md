# 🪖 Helmet Detection using YOLOv8

A deep learning project to detect **helmet** and **no helmet** in images and videos using YOLOv8 object detection model.

---

## 📌 Project Overview

This project uses a custom dataset of ~1180 images (including frames extracted from video) to train a YOLOv8 model to detect whether a person is wearing a helmet or not. This can be used for **road safety monitoring** and **construction site safety**.

---

## 🗂️ Dataset

| Source | Count |
|--------|-------|
| With Helmet Images | ~500 |
| Without Helmet Images | ~320 |
| Extracted from Video | ~360 |
| **Total** | **~1180** |

- Labeled using **Roboflow** (bounding box annotation)
- Exported in **YOLO format**
- Classes: `with_helmet`, `without_helmet`

---

## 🛠️ Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Roboflow (annotation)
- Kaggle (training GPU)

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install ultralytics opencv-python roboflow
```

### 2. Extract Frames from Video
```python
import cv2, os
cap = cv2.VideoCapture('video.mp4')
os.makedirs('frames', exist_ok=True)
count, saved = 0, 0
while True:
    ret, frame = cap.read()
    if not ret: break
    if count % 10 == 0:
        cv2.imwrite(f'frames/frame_{saved}.jpg', frame)
        saved += 1
    count += 1
cap.release()
```

### 3. Train Model
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='helmet.yaml', epochs=50, imgsz=640, batch=16)
```

### 4. Run Inference
```python
model.predict(source='test.jpg', save=True)
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| mAP@50 | _coming soon_ |
| Precision | _coming soon_ |
| Recall | _coming soon_ |

---

## 📁 Project Structure

```
helmet-detection/
├── dataset/
│   ├── images/
│   └── labels/
├── extracted_frames/
├── runs/                  # YOLOv8 training output
├── helmet.yaml
└── train.py
```

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com)
