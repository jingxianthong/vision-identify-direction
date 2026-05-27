# Raspbot Autonomous Emergency Braking (AEB) System

This document outlines the setup and operational logic of the custom Autonomous Emergency Braking (AEB) script for the Yahboom Raspbot. The system utilizes a camera and a custom-trained YOLO object detection model to identify borders or obstacles, adjusting the robot's speed and initiating evasive maneuvers when necessary.

---

## 🎥 Video Demonstration

**Watch the Raspbot AEB system in action here:**
[Insert YouTube Link Here]

---

## ⚙️ System Overview

The script relies on the `ultralytics` YOLO engine and OpenCV to process live video feeds. To optimize performance on the Raspberry Pi, the AI inference is executed **once every 5 frames**. The system estimates the proximity of a detected "Border" based on the pixel width of its bounding box.

### Key Features

* **Custom AI Model:** Uses `best_V02.pt` specifically trained to recognize a `Border` class.
* **Resource Optimization:** AI processing is limited to 20% of total frames to maintain real-time camera responsiveness.
* **Live Visual Feedback:** Displays bounding boxes and status messages on a desktop GUI window (`Raspbot AEB Vision`).
* **Hardware Integration:** Outputs real-time telemetry to the built-in OLED screen and utilizes the active buzzer during emergency stops.

---

## 🧠 AEB Decision Logic

The robot calculates its distance from the target using the width of the detected bounding box (`border_width = x2 - x1`). Based on this width, the system transitions between three distinct driving states:

| Bounding Box Width | System State | Visual UI Color | Motor Action | Speed Output |
| --- | --- | --- | --- | --- |
| **No Detection** | Path Clear | N/A | Normal Driving | `30` |
| **< 300 pixels** | Border Ahead | Green | Approaching | `10` |
| **300 - 449 pixels** | Slowing Down | Yellow | Decelerating | `5` |
| **≥ 450 pixels** | Evading | Red | Brake & Reverse / Horn Beep | `0` (Reverses at `-15`) |

---

## 💻 The Source Code

Below is the complete implementation of the AEB system. Ensure this script is run in an environment where the Yahboom OLED and motor libraries are correctly pathed.

```python
import os
# THESE MUST BE AT THE VERY TOP
os.environ["QT_QPA_PLATFORM"] = "xcb" 
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import time
import sys

print("Opening video window FIRST...")
cv2.namedWindow("Raspbot AEB Vision")
cv2.waitKey(1)

from ultralytics import YOLO 

# Path and Hardware Setup
sys.path.append('/home/pi/software/oled_yahboom/')
from yahboom_oled import *
sys.path.append('/home/pi/project_demo/lib')
from McLumk_Wheel_Sports import *
from Raspbot_Lib import Raspbot

# Initialization
bot = Raspbot()
oled = Yahboom_OLED(debug=False)
oled.init_oled_process()

print("Loading YOLO model... (This takes a moment)")
# Custom brain loaded
model = YOLO('best_V02.pt')
print("AI Classes loaded:", model.names)

# Clear camera locks
os.system("sudo fuser -k /dev/video*")
time.sleep(1) 

# Connect to Camera
print("Connecting to Camera on index 0...")
cap = cv2.VideoCapture('/dev/video0',cv2.CAP_V4L2) 
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("ERROR: Camera failed to open.")
    sys.exit()

bot.Ctrl_Servo(1, 80)
bot.Ctrl_Servo(2, 0) 

print("Autonomous AEB Active. Press 'q' in the VIDEO window to quit.")

frame_count = 0
last_boxes = [] 

try:
    target_found = False
    action_msg = "PATH CLEAR"
    box_color = (0, 255, 0) # Green
    current_speed = 5 # Normal driving speed
    while True:
        ret, frame = cap.read()
        # If the frame is blank, print warning and try again
        if not ret or frame is None: 
            print("Warning: No video frame received.")
            time.sleep(1)
            continue
            
        frame_count += 1
        
        if frame_count == 5:
            print("Processing first AI frame... (Please wait!)")
        
        # --- ONLY RUN AI EVERY 5 FRAMES ---
        if frame_count % 5 == 0:
            target_found = False
            action_msg = "PATH CLEAR"
            box_color = (0, 255, 0) # Green
            current_speed = 10 # Normal driving speed
        
            results = model(frame, conf=0.25, imgsz=480)
            last_boxes = [] 
            
            for r in results:
                for box in r.boxes:
                    label = model.names[int(box.cls[0])]
                    
                    if label == 'Border': 
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        border_width = int(x2 - x1)
                        
                        # DISTANCE LOGIC
                        if border_width < 300:
                            action_msg = "BORDER AHEAD"
                            box_color = (0, 255, 0) #Green
                            current_speed = 10 
                        elif 300 <= border_width < 450:
                            action_msg = "SLOWING DOWN"
                            box_color = (0, 255, 255) #Yellow
                            current_speed = 5 
                        else: 
                            action_msg = "Evading"
                            box_color = (0, 0, 255) #Red
                            current_speed = 0 
                        
                        last_boxes.append((int(x1), int(y1), int(x2), int(y2), action_msg, box_color))
                        target_found = True
                        break
        
        # --- MOTOR CONTROL ---
        if target_found:
            if current_speed == 0:
                move_param_forward(-15,20)
                bot.Ctrl_BEEP_Switch(1) # Beep horn when stopped
                time.sleep(0.05)
                bot.Ctrl_BEEP_Switch(0)
            else:
                # Drive straight at the calculated speed
                move_param_forward(current_speed, 1) 
        else:
            # If no border is seen, just keep driving
            move_param_forward(30, 1) 
        
        # Update OLED
        oled.clear()
        oled.add_line("AEB SYSTEM", 1)
        oled.add_line(f"STS: {action_msg}", 3)
        oled.refresh()

        # --- DRAW BOXES ON EVERY FRAME ---
        for (x1, y1, x2, y2, msg, color) in last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, msg, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update Desktop Window
        cv2.imshow("Raspbot AEB Vision", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    stop_robot()
    cap.release()
    cv2.destroyAllWindows()
    # Restart the default OLED status script upon exit
    os.system("python3 /home/pi/software/oled_yahboom/yahboom_oled.py &")
    del bot

```
