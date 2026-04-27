# vision-identify-arrow-direction

# 🤖 SKILL FILE: Custom YOLOv8 Autonomous Vision Pipeline

**Skill Name:** `train_custom_direction_yolov8`
**Target Hardware:** Cloud GPU (Training) → Edge Device / Raspberry Pi 5 (Inference)
**Objective:** End-to-end pipeline to capture images, locally annotate bounding boxes, format for YOLOv8, train on Google Colab, and deploy for real-time webcam inference.

**Classes Defined:**

* `0: Left`
* `1: Right`
* `2: Forward`

---

## 🛠️ Phase 1: Local Data Generation & Annotation

**Environment:** Windows Local Machine

**Dependencies:**

```bash
pip install opencv-python
```

### 📄 File: `local_annotator.py`

```python
import cv2
import os
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
IMG_FOLDER = os.path.join(script_dir, "dataset", "images")
LBL_FOLDER = os.path.join(script_dir, "dataset", "labels")
os.makedirs(IMG_FOLDER, exist_ok=True)
os.makedirs(LBL_FOLDER, exist_ok=True)

CLASSES = {ord('1'): 0, ord('2'): 1, ord('3'): 2}

drawing = False
ix, iy, x2, y2 = -1, -1, -1, -1
boxes = []

def draw_box(event, x, y, flags, param):
    global ix, iy, x2, y2, drawing, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True; ix, iy = x, y; x2, y2 = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        x2, y2 = x, y
        img_copy = clone.copy()
        cv2.rectangle(img_copy, (ix, iy), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Custom Annotator", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False; x2, y2 = x, y
        cv2.rectangle(clone, (ix, iy), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Custom Annotator", clone)
        boxes.append((min(ix, x2), min(iy, y2), max(ix, x2), max(iy, y2)))

raw_images_dir = "raw_photos"
images = [f for f in os.listdir(raw_images_dir) if f.lower().endswith(('.png', '.jpg'))]

cv2.namedWindow("Custom Annotator")
cv2.setMouseCallback("Custom Annotator", draw_box)

for image_name in images:
    img_path = os.path.join(raw_images_dir, image_name)
    clone = cv2.imread(img_path)
    img_h, img_w = clone.shape[:2]
    boxes = []
    cv2.imshow("Custom Annotator", clone)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in CLASSES and len(boxes) > 0:
            class_id = CLASSES[key]
            x_min, y_min, x_max, y_max = boxes[-1]

            x_center = ((x_min + x_max) / 2) / img_w
            y_center = ((y_min + y_max) / 2) / img_h
            box_w = (x_max - x_min) / img_w
            box_h = (y_max - y_min) / img_h

            txt_name = os.path.splitext(image_name)[0] + ".txt"
            with open(os.path.join(LBL_FOLDER, txt_name), "w") as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

            shutil.copy(img_path, os.path.join(IMG_FOLDER, image_name))
            break

        elif key == ord('c'):
            clone = cv2.imread(img_path)
            boxes = []
            cv2.imshow("Custom Annotator", clone)

        elif key == ord('s'):
            break

        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
```

**Action:** Zip the dataset folder → `full_dataset.zip`

---

## ☁️ Phase 2: Cloud Processing & YOLO Formatting

**Environment:** Google Colab (T4 GPU Runtime)

### 📄 File: `colab_prep.py`

```python
import os, shutil, glob, zipfile

shutil.rmtree('/content/dataset', ignore_errors=True)
os.makedirs('/content/dataset/images', exist_ok=True)
os.makedirs('/content/dataset/labels', exist_ok=True)

zip_files = glob.glob('/content/*.zip')

if zip_files:
    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
        zip_ref.extractall('/content/temp_extract')

    img_count, lbl_count = 0, 0

    for root, dirs, files in os.walk('/content/temp_extract'):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.move(os.path.join(root, f), '/content/dataset/images/' + f)
                img_count += 1
            elif f.lower().endswith('.txt') and f not in ['classes.txt', 'data.yaml']:
                shutil.move(os.path.join(root, f), '/content/dataset/labels/' + f)
                lbl_count += 1

    print(f"✅ Ready! Images: {img_count} | Labels: {lbl_count}")
```

---

## 🧠 Phase 3: Model Training

**Dependencies:**

```bash
pip install ultralytics
```

### 📄 File: `colab_train.py`

```python
import yaml
from ultralytics import YOLO

yaml_data = {
    'train': '/content/dataset/images',
    'val': '/content/dataset/images',
    'nc': 3,
    'names': ['Left', 'Right', 'Forward']
}

with open('/content/data.yaml', 'w') as f:
    yaml.dump(yaml_data, f)

model = YOLO('yolov8n.pt')
results = model.train(data='/content/data.yaml', epochs=60, imgsz=480, batch=16)

print("✅ Training complete. Download best.pt")
```

---

## 🚀 Phase 4: Edge Device Inference

**Environment:** Local Machine / Raspberry Pi

**Dependencies:**

```bash
pip install ultralytics opencv-python
```

### 📄 File: `live_inference.py`

```python
import cv2
from ultralytics import YOLO

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    results = model(frame, stream=True, conf=0.5)

    for r in results:
        annotated_frame = r.plot()

    cv2.imshow("Vision System - Live Feed", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## ✅ Final Workflow Summary

1. Capture & annotate images locally
2. Zip dataset → upload to Colab
3. Train YOLOv8 model on GPU
4. Download `best.pt`
5. Deploy on Raspberry Pi for real-time inference

---

## 📌 Notes

* Use **YOLOv8n** for best performance on edge devices
* Increase epochs if dataset is small
* Adjust `conf` threshold depending on accuracy vs responsiveness
* Consider adding data augmentation for better robustness

---

🚀 You now have a complete reproducible vision pipeline for your robot.
