import cv2
import time

cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
cap.set(3, 640)
cap.set(4, 480)

count = 0
print("Press 'Space' to take a photo. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    cv2.imshow("Data Collection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '): # Spacebar
        cv2.imwrite(f"border _{count}.jpg", frame)
        print(f"Saved border_{count}.jpg")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()