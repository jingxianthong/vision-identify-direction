import cv2
import os

# --- CONFIGURATION ---
# Put all your photos in a folder named 'dataset' next to this script
script_dir = os.path.dirname(os.path.abspath(__file__))

IMAGE_FOLDER = os.path.join(script_dir, "dataset", "border")
# NEW: This is where the text files will be saved!
LABEL_FOLDER = os.path.join(script_dir, "dataset", "rCircle_border_photo_data") 

# Your AI Classes (Updated for "Border")
CLASSES = {
    ord('1'): 0,  # Press '1' for Border
}

# --- DISPLAY SIZE (change these if the window is still too big/small) ---
DISPLAY_MAX_WIDTH  = 900
DISPLAY_MAX_HEIGHT = 700
# ---------------------

drawing = False
ix, iy, x2, y2 = -1, -1, -1, -1
clone = None        # resized display image
orig = None         # original full-size image
scale_x = 1.0      # ratio: original / display
scale_y = 1.0
boxes = []          # boxes stored in ORIGINAL image coordinates

def draw_box(event, x, y, flags, param):
    global ix, iy, x2, y2, drawing, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        x2, y2 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x2, y2 = x, y
            img_copy = clone.copy()
            cv2.rectangle(img_copy, (ix, iy), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Custom Annotator", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y
        cv2.rectangle(clone, (ix, iy), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Custom Annotator", clone)
        # Scale display coordinates back to original image coordinates before saving
        ox1 = int(min(ix, x2) * scale_x)
        oy1 = int(min(iy, y2) * scale_y)
        ox2 = int(max(ix, x2) * scale_x)
        oy2 = int(max(iy, y2) * scale_y)
        boxes.append((ox1, oy1, ox2, oy2))
        print("Box drawn! Press 1 (Border) to save.")

# Check if folder exists
if not os.path.exists(IMAGE_FOLDER):
    print(f"Error: Could not find the '{IMAGE_FOLDER}' folder.")
    exit()

# Get all images
images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

cv2.namedWindow("Custom Annotator")
cv2.setMouseCallback("Custom Annotator", draw_box)

print("--- CONTROLS ---")
print("Mouse : Click and drag to draw a box")
print("1     : Save box as 'Border' and move to next image")
print("D     : Delete/Undo the last box you drew")
print("C     : Clear all boxes on the image")
print("S     : Skip this image")
print("Q     : Quit the program")
print("----------------")

for image_name in images:
    img_path = os.path.join(IMAGE_FOLDER, image_name)
    orig  = cv2.imread(img_path)
    img_h, img_w = orig.shape[:2]
    boxes = []

    # --- Scale down for display only ---
    scale_x = img_w / min(img_w, DISPLAY_MAX_WIDTH)
    scale_y = img_h / min(img_h, DISPLAY_MAX_HEIGHT)
    scale   = max(scale_x, scale_y)          # keep aspect ratio
    scale_x = scale
    scale_y = scale
    disp_w  = int(img_w / scale)
    disp_h  = int(img_h / scale)
    clone   = cv2.resize(orig, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    # ------------------------------------

    cv2.imshow("Custom Annotator", clone)

    while True:
        key = cv2.waitKey(1) & 0xFF

        # If you pressed 1 and a box was drawn
        # If you pressed 1 and boxes were drawn
        if key in CLASSES and len(boxes) > 0:
            class_id = CLASSES[key]
            
            txt_name = os.path.splitext(image_name)[0] + ".txt"
            txt_path = os.path.join(LABEL_FOLDER, txt_name)
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            
            # --- THE FIX: LOOP THROUGH ALL BOXES ---
            with open(txt_path, "w") as f:
                for box in boxes:
                    x_min, y_min, x_max, y_max = box
                    
                    # Convert to YOLO format (Normalized center points)
                    x_center = ((x_min + x_max) / 2) / img_w
                    y_center = ((y_min + y_max) / 2) / img_h
                    box_w = (x_max - x_min) / img_w
                    box_h = (y_max - y_min) / img_h
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
            
            print(f"Saved {len(boxes)} boxes -> {txt_path}")
            break # Move to the next picture # Move to the next picture

        elif key == ord('d'): # D to Delete/Undo last box
            if len(boxes) > 0:
                boxes.pop() # Remove the last box from memory
                clone = cv2.resize(cv2.imread(img_path), (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                # Redraw remaining boxes (convert back to display coords)
                for b in boxes:
                    dx1, dy1 = int(b[0]/scale_x), int(b[1]/scale_y)
                    dx2, dy2 = int(b[2]/scale_x), int(b[3]/scale_y)
                    cv2.rectangle(clone, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                cv2.imshow("Custom Annotator", clone)
                print("Deleted the last box!")
            else:
                print("Nothing to delete.")

        elif key == ord('c'): # C to Clear ALL boxes
            clone = cv2.resize(cv2.imread(img_path), (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            boxes = []
            cv2.imshow("Custom Annotator", clone)
            print("Cleared all boxes. Draw again.")
            
        elif key == ord('s'): # S to Skip
            print("Skipped image.")
            break
            
        elif key == ord('q'): # Q to Quit
            print("Quitting program.")
            cv2.destroyAllWindows()
            exit()
            
cv2.destroyAllWindows()
print("All images annotated! You are ready for Google Colab.")