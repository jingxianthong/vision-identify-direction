import cv2
import time

# Notice we removed the QT_QPA_PLATFORM line to see if that was crashing the window!

print("Connecting to camera 0...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera failed to open.")
    exit()

print("Camera opened! Starting loop...")
count = 0

while True:
    count += 1
    print(f"Asking camera for frame {count}...")
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
        
    print(f"Successfully grabbed frame {count}! Drawing window...")
    cv2.imshow("Test Window", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()