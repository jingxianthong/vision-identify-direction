#border checker yalo8
import yaml
import os # Added import for os module
from ultralytics import YOLO

# 1. Create the YOLO map
yaml_data = {
    'train': '/content/dataset/images',
    'val': '/content/dataset/images',
    'nc': 1,               # CHANGED: Number of classes is now 1
    'names': ['Border']    # CHANGED: Name of your single class
}

# Save the map to Colab
with open('/content/data.yaml', 'w') as f:
    yaml.dump(yaml_data, f)

# Get the image directory path from yaml_data
IMG_DIR = yaml_data['train']

# Add a check to verify if the image directory exists and is not empty
if not os.path.exists(IMG_DIR):
    print(f"❌ ERROR: The image directory '{IMG_DIR}' does not exist. Please check your data preparation step.")
elif not os.listdir(IMG_DIR):
    print(f"❌ ERROR: The image directory '{IMG_DIR}' is empty. No images found for training. Please check your data preparation step.")
else:
    print(f"✅ Image directory '{IMG_DIR}' found with {len(os.listdir(IMG_DIR))} items. Proceeding with training.")
    # 2. Train the robot's new brain!
    model = YOLO('yolov8n.pt')
    results = model.train(data='/content/data.yaml', epochs=50, imgsz=480)