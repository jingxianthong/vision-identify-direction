import cv2
import os
import sys

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))

IMAGE_FOLDER = os.path.join(script_dir, "dataset", "border")
LABEL_FOLDER = os.path.join(script_dir, "dataset", "rCircle_border_photo_data")

DISPLAY_MAX_WIDTH  = 900
DISPLAY_MAX_HEIGHT = 700

CLASS_NAMES = { 0: "Border" }

# --- Arrow key codes (Windows 11) ---
KEY_LEFT  = 2424832
KEY_RIGHT = 2555904
KEY_UP    = 2490368
KEY_DOWN  = 2621440

# --- Dropdown UI layout (in display pixels) ---
BTN_X, BTN_Y, BTN_W, BTN_H = 5, 5, 220, 30   # the button
ITEM_H       = 26                               # height of each list row
ITEMS_VISIBLE = 10                              # max rows shown at once
# ----------------------------------------------

if not os.path.exists(IMAGE_FOLDER) or not os.path.exists(LABEL_FOLDER):
    print("Error: Could not find image or label folders.")
    sys.exit()

images = sorted([f for f in os.listdir(IMAGE_FOLDER)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
if not images:
    print("No images found.")
    sys.exit()

# --- Shared state ---
state = {
    'i':            0,        # current image index
    'dropdown':     False,    # is dropdown open?
    'scroll':       0,        # first visible item index in dropdown
    'click_x':      -1,
    'click_y':      -1,
    'clicked':      False,
    'scroll_delta': 0,
    'navigate':     0,        # +1 forward, -1 backward, 0 nothing
    'quit':         False,
}

# -----------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------

def draw_button(canvas, index, total, is_open):
    """Draw the dropdown button on the canvas."""
    # Shadow / background
    cv2.rectangle(canvas,
                  (BTN_X, BTN_Y),
                  (BTN_X + BTN_W, BTN_Y + BTN_H),
                  (40, 40, 40), -1)
    cv2.rectangle(canvas,
                  (BTN_X, BTN_Y),
                  (BTN_X + BTN_W, BTN_Y + BTN_H),
                  (120, 120, 120), 1)

    # Label text
    label = f"Image {index + 1} / {total}"
    cv2.putText(canvas, label,
                (BTN_X + 8, BTN_Y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # Arrow indicator  ▼ or ▲
    arrow = "A" if is_open else "V"   # ASCII stand-in for ▲ / ▼
    cv2.putText(canvas, arrow,
                (BTN_X + BTN_W - 20, BTN_Y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)


def draw_dropdown(canvas, image_list, current_idx, scroll_top):
    """Draw the open dropdown list below the button."""
    list_x = BTN_X
    list_y = BTN_Y + BTN_H
    list_w = BTN_W + 80   # a bit wider to show full filenames

    visible = image_list[scroll_top: scroll_top + ITEMS_VISIBLE]
    list_h  = len(visible) * ITEM_H + 4

    # Background panel
    cv2.rectangle(canvas,
                  (list_x, list_y),
                  (list_x + list_w, list_y + list_h),
                  (30, 30, 30), -1)
    cv2.rectangle(canvas,
                  (list_x, list_y),
                  (list_x + list_w, list_y + list_h),
                  (100, 100, 100), 1)

    for row, name in enumerate(visible):
        abs_idx = scroll_top + row
        item_y1 = list_y + row * ITEM_H + 2
        item_y2 = item_y1 + ITEM_H - 2

        # Highlight current image
        bg_color = (60, 80, 120) if abs_idx == current_idx else (30, 30, 30)
        cv2.rectangle(canvas, (list_x + 1, item_y1),
                      (list_x + list_w - 1, item_y2), bg_color, -1)

        text  = f"{abs_idx + 1:>3}.  {name}"
        color = (0, 200, 255) if abs_idx == current_idx else (200, 200, 200)
        cv2.putText(canvas, text,
                    (list_x + 6, item_y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Scroll hint at bottom if list is long
    if len(image_list) > ITEMS_VISIBLE:
        hint = f"scroll  [{scroll_top + 1}–{min(scroll_top + ITEMS_VISIBLE, len(image_list))} of {len(image_list)}]"
        cv2.putText(canvas, hint,
                    (list_x + 4, list_y + list_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    return list_w, list_h   # return size for hit testing


def is_in_button(x, y):
    return BTN_X <= x <= BTN_X + BTN_W and BTN_Y <= y <= BTN_Y + BTN_H


def item_clicked(x, y, scroll_top, list_w, list_h, total):
    """Return the clicked item index, or -1 if outside list."""
    list_x = BTN_X
    list_y = BTN_Y + BTN_H
    if list_x <= x <= list_x + list_w and list_y <= y <= list_y + list_h:
        row = (y - list_y) // ITEM_H
        idx = scroll_top + row
        if 0 <= idx < total:
            return idx
    return -1

# -----------------------------------------------------------------------
# Mouse callback
# -----------------------------------------------------------------------

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state['click_x'] = x
        state['click_y'] = y
        state['clicked'] = True
    elif event == cv2.EVENT_MOUSEWHEEL:
        state['scroll_delta'] = 1 if flags > 0 else -1

# -----------------------------------------------------------------------
# Build annotated image
# -----------------------------------------------------------------------

def build_frame(img_path, txt_path, img_w, img_h):
    img = cv2.imread(img_path)
    if img is None:
        return None

    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    xc = float(parts[1]); yc = float(parts[2])
                    bw = float(parts[3]); bh = float(parts[4])
                    x1 = int((xc - bw / 2) * img_w)
                    y1 = int((yc - bh / 2) * img_h)
                    x2 = int((xc + bw / 2) * img_w)
                    y2 = int((yc + bh / 2) * img_h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    lname = CLASS_NAMES.get(class_id, f"Class {class_id}")
                    cv2.putText(img, lname, (x1, max(y1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        cv2.putText(img, "NO TAGS FOUND", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Filename at bottom
    cv2.putText(img, images[state['i']], (10, img_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return img

# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------

cv2.namedWindow("Dataset Verification")
cv2.setMouseCallback("Dataset Verification", on_mouse)

print("--- CONTROLS ---")
print("← / →  Arrow keys or A/D : Navigate")
print("Click the dropdown button : Jump to any image")
print("Q                         : Quit")
print("----------------")

last_list_w = BTN_W + 80
last_list_h = ITEMS_VISIBLE * ITEM_H

while state['i'] < len(images) and not state['quit']:
    i        = state['i']
    img_path = os.path.join(IMAGE_FOLDER, images[i])
    txt_path = os.path.join(LABEL_FOLDER, os.path.splitext(images[i])[0] + ".txt")

    raw = cv2.imread(img_path)
    if raw is None:
        state['i'] += 1
        continue
    img_h, img_w = raw.shape[:2]

    # Scale for display
    scale_x = img_w / min(img_w, DISPLAY_MAX_WIDTH)
    scale_y = img_h / min(img_h, DISPLAY_MAX_HEIGHT)
    scale   = max(scale_x, scale_y)
    disp_w  = int(img_w / scale)
    disp_h  = int(img_h / scale)

    base = build_frame(img_path, txt_path, img_w, img_h)
    if base is None:
        state['i'] += 1
        continue

    display = cv2.resize(base, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    # Draw UI on top of display
    draw_button(display, i, len(images), state['dropdown'])
    if state['dropdown']:
        last_list_w, last_list_h = draw_dropdown(
            display, images, i, state['scroll'])

    cv2.imshow("Dataset Verification", display)

    # --- Process mouse scroll ---
    if state['scroll_delta'] != 0:
        if state['dropdown']:
            state['scroll'] = max(0, min(
                state['scroll'] - state['scroll_delta'],
                len(images) - ITEMS_VISIBLE
            ))
        state['scroll_delta'] = 0

    # --- Process mouse click ---
    if state['clicked']:
        cx, cy = state['click_x'], state['click_y']
        state['clicked'] = False

        if state['dropdown']:
            # Check if clicked inside the list
            chosen = item_clicked(cx, cy, state['scroll'],
                                  last_list_w, last_list_h, len(images))
            if chosen >= 0:
                state['i'] = chosen
                state['dropdown'] = False
            elif is_in_button(cx, cy):
                state['dropdown'] = False   # toggle close
            else:
                state['dropdown'] = False   # click outside — close
        else:
            if is_in_button(cx, cy):
                # Open dropdown, scroll so current item is visible
                state['scroll'] = max(0, min(
                    i,
                    len(images) - ITEMS_VISIBLE
                ))
                state['dropdown'] = True

    # --- Process key ---
    key = cv2.waitKeyEx(1)

    if key == ord('q') or key == ord('Q'):
        state['quit'] = True

    elif key in (KEY_LEFT, ord('a'), ord('A'), ord('p'), ord('P')):
        state['dropdown'] = False
        state['i'] = max(0, state['i'] - 1)

    elif key in (KEY_RIGHT, KEY_UP, ord('d'), ord('D'),
                 ord('n'), ord('N'), ord(' ')):
        state['dropdown'] = False
        state['i'] += 1

cv2.destroyAllWindows()
print("Verification complete!")