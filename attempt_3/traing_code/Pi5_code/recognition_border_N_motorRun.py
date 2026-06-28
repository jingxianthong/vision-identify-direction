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
model = YOLO('best_V03.pt')
print("AI Classes loaded:", model.names)

# Clear camera locks
os.system("sudo fuser -k /dev/video*")
time.sleep(1) 

# Connect to Camera
print("Connecting to Camera on index 0...")
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2) 
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("ERROR: Camera failed to open.")
    sys.exit()

bot.Ctrl_Servo(1, 80)
bot.Ctrl_Servo(2, 10) 

print("Autonomous AEB Active. Press 'q' in the VIDEO window to quit.")

frame_count = 0
last_boxes = [] 

try:
    target_found = False
    action_msg = "PATH CLEAR"
    current_speed = 15 # Default normal driving speed
    
    while True:
        ret, frame = cap.read()
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
            current_speed = 15 # Reset to normal speed before frame re-evaluation
            last_boxes = [] 
        
            results = model(frame, conf=0.25, imgsz=480)
            
            for r in results:
                for box in r.boxes:
                    label = model.names[int(box.cls[0])]
                    
                    if label == 'Border': 
                        target_found = True
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Use bottom edge (y2) for true distance. Frame height is 480.
                        bottom_edge = int(y2)
                        
                        # Calculate individual threat parameters for this bounding box
                        if bottom_edge < 200:
                            local_msg = "BORDER AHEAD"
                            local_color = (0, 255, 0) # Green
                            local_speed = 10 
                        elif 200 <= bottom_edge < 290:
                            local_msg = "SLOWING DOWN"
                            local_color = (0, 255, 255) # Yellow
                            local_speed = 5 
                        else: 
                            local_msg = "EVADING!"
                            local_color = (0, 0, 255) # Red
                            local_speed = 0 
                        
                        # APPEND ALL BOXES (Removed 'break' so everything draws simultaneously)
                        last_boxes.append((int(x1), int(y1), int(x2), int(y2), local_msg, local_color))
                        
                        # GLOBAL THREAT ASSESSMENT
                        # Prioritize the lowest speed restriction (highest collision threat)
                        if local_speed < current_speed:
                            current_speed = local_speed
                            action_msg = local_msg
        
        # --- MOTOR CONTROL WITH CORNER ESCAPE SEQUENCE ---
        if target_found:
            if current_speed == 0:
                # Immediate brakes and warning audio
                stop_robot()
                bot.Ctrl_BEEP_Switch(1) 
                time.sleep(0.1)
                bot.Ctrl_BEEP_Switch(0)
                
                # Active Escape Sequence: Back away and rotate out of the corner completely
                print("DANGER: Executing Committed Escape Maneuver!")
                move_backward(15)
                time.sleep(0.7) # Reverse distance duration
                rotate_right(20)
                time.sleep(0.6) # Turn angle duration
                stop_robot()
                
                # Flash states to clean buffers before next read
                target_found = False
                # last_boxes = []
                frame_count = 0
            else:
                # Move forward at the dynamically assigned safe speed
                move_param_forward(current_speed, 1) 
        else:
            # Cruise normally if no borders are visible
            move_param_forward(15, 1) 
        
        # Update OLED Display
        oled.clear()
        oled.add_line("AEB SYSTEM", 1)
        oled.add_line(f"STS: {action_msg}", 3)
        oled.refresh()

        # --- DRAW ALL VALIDATED BOUNDING BOXES ON SCREEN ---
        for (x1, y1, x2, y2, msg, color) in last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, msg, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update Desktop Window view
        cv2.imshow("Raspbot AEB Vision", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    stop_robot()
    cap.release()
    cv2.destroyAllWindows()
    os.system("python3 /home/pi/software/oled_yahboom/yahboom_oled.py &")
    del bot