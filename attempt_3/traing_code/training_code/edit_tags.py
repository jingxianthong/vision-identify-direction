import cv2
import os
import sys

# 1. Environment fix for Raspberry Pi 5 Display
os.environ["QT_QPA_PLATFORM"] = "xcb"

# --- CONFIGURATION ---
script_dir   = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(script_dir, "dataset", "border")
LABEL_FOLDER = os.path.join(script_dir, "dataset", "rCircle_border_photo_data")

DISPLAY_MAX_WIDTH  = 900
DISPLAY_MAX_HEIGHT = 700

# --- Dropdown UI layout (display pixels) ---
BTN_X, BTN_Y, BTN_W, BTN_H = 5, 5, 220, 30
ITEM_H        = 26
ITEMS_VISIBLE = 10
# -------------------------------------------

# -----------------------------------------------------------------------
# Shared state
# -----------------------------------------------------------------------
state = {
    'i':           0,
    'dropdown':    False,
    'scroll':      0,
    'list_w':      BTN_W + 80,
    'list_h':      ITEMS_VISIBLE * ITEM_H,
    'result':      None,   # 'next' | 'skip' | 'quit' | int (jump index)
    'scroll_delta': 0,
}

# Drawing state (display coordinates)
draw_state = {
    'active': False,
    'ix': -1, 'iy': -1,
    'x2': -1, 'y2': -1,
}

# Per-image globals (set when loading each image)
orig    = None   # original full-size image
scale_x = 1.0
scale_y = 1.0
disp_w  = 0
disp_h  = 0
img_w   = 0
img_h   = 0
boxes   = []     # stored in ORIGINAL image coordinates
images  = []

# -----------------------------------------------------------------------
# Coordinate helpers
# -----------------------------------------------------------------------
def to_disp(ox, oy):
    """Original pixel → display pixel."""
    return int(ox / scale_x), int(oy / scale_y)

def to_orig(dx, dy):
    """Display pixel → original pixel."""
    return int(dx * scale_x), int(dy * scale_y)

# -----------------------------------------------------------------------
# Dropdown drawing helpers  (identical to verify_tags.py)
# -----------------------------------------------------------------------
# Tag status cache — built once, updated only on save/delete (fast O(1) lookup)
# Populated after `images` is loaded (see bottom of config section)
tagged_set: set = set()

def has_label(image_name):
    """O(1) cache lookup — no filesystem access per frame."""
    return image_name in tagged_set

def mark_tagged(image_name):
    tagged_set.add(image_name)

def mark_untagged(image_name):
    tagged_set.discard(image_name)

def count_untagged(image_list):
    return sum(1 for n in image_list if n not in tagged_set)


def draw_button(canvas, index, total, is_open):
    untagged = count_untagged(images)

    # Background — orange tint if there are still untagged images
    bg_color  = (30, 55, 80) if untagged == 0 else (60, 40, 20)
    bdr_color = (0, 200, 100) if untagged == 0 else (0, 140, 255)

    cv2.rectangle(canvas, (BTN_X, BTN_Y),
                  (BTN_X + BTN_W, BTN_Y + BTN_H), bg_color, -1)
    cv2.rectangle(canvas, (BTN_X, BTN_Y),
                  (BTN_X + BTN_W, BTN_Y + BTN_H), bdr_color, 1)

    label = f"Image {index + 1} / {total}"
    cv2.putText(canvas, label, (BTN_X + 8, BTN_Y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    # Untagged counter on the right side of the button
    if untagged > 0:
        tag_txt = f"{untagged} left"
        cv2.putText(canvas, tag_txt,
                    (BTN_X + BTN_W + 5, BTN_Y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "All done!",
                    (BTN_X + BTN_W + 5, BTN_Y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 100), 1, cv2.LINE_AA)

    arrow = "A" if is_open else "V"
    cv2.putText(canvas, arrow, (BTN_X + BTN_W - 20, BTN_Y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)


def draw_dropdown(canvas, image_list, current_idx, scroll_top):
    list_x = BTN_X
    list_y = BTN_Y + BTN_H
    list_w = BTN_W + 120   # wider to fit badge
    visible = image_list[scroll_top: scroll_top + ITEMS_VISIBLE]
    list_h  = len(visible) * ITEM_H + 4

    cv2.rectangle(canvas, (list_x, list_y),
                  (list_x + list_w, list_y + list_h), (30, 30, 30), -1)
    cv2.rectangle(canvas, (list_x, list_y),
                  (list_x + list_w, list_y + list_h), (100, 100, 100), 1)

    for row, name in enumerate(visible):
        abs_idx  = scroll_top + row
        tagged   = has_label(name)
        iy1 = list_y + row * ITEM_H + 2
        iy2 = iy1 + ITEM_H - 2

        # Row background:
        # current image  → blue highlight
        # untagged       → dark red tint
        # tagged         → normal dark
        if abs_idx == current_idx:
            bg = (60, 80, 120)
        elif not tagged:
            bg = (40, 20, 20)   # dark red tint for untagged
        else:
            bg = (30, 30, 30)

        cv2.rectangle(canvas, (list_x + 1, iy1),
                      (list_x + list_w - 1, iy2), bg, -1)

        # Filename text
        text_color = (0, 200, 255) if abs_idx == current_idx else (200, 200, 200)
        cv2.putText(canvas, f"{abs_idx + 1:>3}.  {name}",
                    (list_x + 6, iy1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        # Badge on the right side
        badge_x = list_x + list_w - 62
        if tagged:
            # Green  OK  badge
            cv2.rectangle(canvas, (badge_x, iy1 + 3),
                          (badge_x + 52, iy2 - 3), (20, 80, 20), -1)
            cv2.putText(canvas, "OK",
                        (badge_x + 14, iy1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 80), 1, cv2.LINE_AA)
        else:
            # Red  NO TAG  badge
            cv2.rectangle(canvas, (badge_x, iy1 + 3),
                          (badge_x + 52, iy2 - 3), (80, 20, 20), -1)
            cv2.putText(canvas, "NO TAG",
                        (badge_x + 2, iy1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 255), 1, cv2.LINE_AA)

    if len(image_list) > ITEMS_VISIBLE:
        hint = f"scroll  [{scroll_top+1}-{min(scroll_top+ITEMS_VISIBLE, len(image_list))} of {len(image_list)}]"
        cv2.putText(canvas, hint, (list_x + 4, list_y + list_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    state['list_w'] = list_w
    state['list_h'] = list_h


def is_in_button(x, y):
    return BTN_X <= x <= BTN_X + BTN_W and BTN_Y <= y <= BTN_Y + BTN_H


def item_at(x, y, scroll_top):
    list_x = BTN_X
    list_y = BTN_Y + BTN_H
    lw = state['list_w']
    lh = state['list_h']
    if list_x <= x <= list_x + lw and list_y <= y <= list_y + lh:
        row = (y - list_y) // ITEM_H
        idx = scroll_top + row
        if 0 <= idx < len(images):
            return idx
    return -1

# -----------------------------------------------------------------------
# Screen refresh  — always redraws from `orig`
# -----------------------------------------------------------------------
def refresh_screen(dragging_box=None):
    """
    Redraws the display frame: scaled image + boxes + dropdown UI.
    dragging_box: optional (x1,y1,x2,y2) in DISPLAY coords drawn in red.
    """
    display = cv2.resize(orig, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    # Saved boxes in green (convert original → display)
    for b in boxes:
        dx1, dy1 = to_disp(b[0], b[1])
        dx2, dy2 = to_disp(b[2], b[3])
        cv2.rectangle(display, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)

    # Box being drawn right now in red
    if dragging_box:
        cv2.rectangle(display,
                      (dragging_box[0], dragging_box[1]),
                      (dragging_box[2], dragging_box[3]),
                      (0, 0, 255), 2)

    # Progress counter
    cv2.putText(display, f"{state['i']+1} / {len(images)}",
                (disp_w - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Filename at bottom
    cv2.putText(display, images[state['i']], (10, disp_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Key hints (bottom-right area)
    hints = ["1=Save  D=Undo  C=Clear  S=Skip  Q=Quit"]
    cv2.putText(display, hints[0], (10, disp_h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    # Dropdown button + list
    draw_button(display, state['i'], len(images), state['dropdown'])
    if state['dropdown']:
        draw_dropdown(display, images, state['i'], state['scroll'])

    cv2.imshow("Tag Editor", display)

# -----------------------------------------------------------------------
# Mouse callback  — handles BOTH dropdown clicks AND box drawing
# -----------------------------------------------------------------------
def on_mouse(event, x, y, flags, param):
    global boxes

    # ---- SCROLL WHEEL ----
    if event == cv2.EVENT_MOUSEWHEEL:
        if state['dropdown']:
            state['scroll_delta'] = 1 if flags > 0 else -1
        return

    # ---- LEFT BUTTON DOWN ----
    if event == cv2.EVENT_LBUTTONDOWN:

        # Priority 1: dropdown is open → handle dropdown interaction only
        if state['dropdown']:
            chosen = item_at(x, y, state['scroll'])
            if chosen >= 0:
                state['result'] = chosen   # jump to image
            state['dropdown'] = False
            refresh_screen()
            return

        # Priority 2: clicked the button → open dropdown
        if is_in_button(x, y):
            state['scroll']   = max(0, min(state['i'],
                                           len(images) - ITEMS_VISIBLE))
            state['dropdown'] = True
            refresh_screen()
            return

        # Priority 3: start drawing a new box
        draw_state['active'] = True
        draw_state['ix'] = x
        draw_state['iy'] = y
        draw_state['x2'] = x
        draw_state['y2'] = y

    # ---- MOUSE MOVE ----
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw_state['active']:
            draw_state['x2'] = x
            draw_state['y2'] = y
            refresh_screen(dragging_box=(draw_state['ix'], draw_state['iy'],
                                         x, y))

    # ---- LEFT BUTTON UP ----
    elif event == cv2.EVENT_LBUTTONUP:
        if draw_state['active']:
            draw_state['active'] = False
            draw_state['x2'] = x
            draw_state['y2'] = y

            # Convert display coords → original image coords before saving
            ox1, oy1 = to_orig(min(draw_state['ix'], x),
                                min(draw_state['iy'], y))
            ox2, oy2 = to_orig(max(draw_state['ix'], x),
                                max(draw_state['iy'], y))

            # Ignore tiny accidental clicks
            if abs(ox2 - ox1) > 5 and abs(oy2 - oy1) > 5:
                boxes.append((ox1, oy1, ox2, oy2))
                print(f"Box added! Total boxes: {len(boxes)}")

            refresh_screen()

# -----------------------------------------------------------------------
# Folder / image checks
# -----------------------------------------------------------------------
if not os.path.exists(IMAGE_FOLDER) or not os.path.exists(LABEL_FOLDER):
    print("Error: Could not find the image or label folders.")
    sys.exit()

images = sorted([f for f in os.listdir(IMAGE_FOLDER)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if not images:
    print("No images found.")
    sys.exit()

# --- Build tag cache once (one filesystem scan at startup) ---
for _name in images:
    _txt = os.path.join(LABEL_FOLDER, os.path.splitext(_name)[0] + ".txt")
    if os.path.exists(_txt) and os.path.getsize(_txt) > 0:
        tagged_set.add(_name)

cv2.namedWindow("Tag Editor")
cv2.setMouseCallback("Tag Editor", on_mouse)

print("--- EDITOR CONTROLS ---")
print("Dropdown button    : Jump to any image")
print("Mouse drag         : Draw a new box")
print("1                  : Save & next image")
print("D                  : Undo last box")
print("C                  : Clear all boxes")
print("S                  : Skip (keep original tags)")
print("Q                  : Quit")
print("-----------------------")

# -----------------------------------------------------------------------
# Main loop  (index-based so we can jump to any image)
# -----------------------------------------------------------------------
i = 0
while i < len(images):
    image_name = images[i]
    img_path   = os.path.join(IMAGE_FOLDER, image_name)
    txt_path   = os.path.join(LABEL_FOLDER,
                              os.path.splitext(image_name)[0] + ".txt")

    # Load original
    orig = cv2.imread(img_path)
    if orig is None:
        i += 1
        continue

    img_h, img_w = orig.shape[:2]

    # Compute display scale
    sx = img_w / min(img_w, DISPLAY_MAX_WIDTH)
    sy = img_h / min(img_h, DISPLAY_MAX_HEIGHT)
    sc = max(sx, sy)
    scale_x = sc
    scale_y = sc
    disp_w  = int(img_w / sc)
    disp_h  = int(img_h / sc)

    # Load existing YOLO tags → convert to original pixel coords
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    xc, yc, bw, bh = map(float, parts[1:])
                    x1 = int((xc - bw / 2) * img_w)
                    y1 = int((yc - bh / 2) * img_h)
                    x2 = int((xc + bw / 2) * img_w)
                    y2 = int((yc + bh / 2) * img_h)
                    boxes.append((x1, y1, x2, y2))
        print(f"Loaded {len(boxes)} existing box(es) for {image_name}")

    # Reset per-image state
    state['i']        = i
    state['result']   = None
    state['dropdown'] = False
    draw_state['active'] = False

    refresh_screen()

    # ---- Edit loop for this image ----
    while state['result'] is None:

        # Scroll wheel
        if state['scroll_delta'] != 0:
            if state['dropdown']:
                state['scroll'] = max(0, min(
                    state['scroll'] - state['scroll_delta'],
                    len(images) - ITEMS_VISIBLE
                ))
                refresh_screen()
            state['scroll_delta'] = 0

        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            # Save boxes to txt file
            with open(txt_path, "w") as f:
                for bx in boxes:
                    x_min, y_min, x_max, y_max = bx
                    xc = ((x_min + x_max) / 2) / img_w
                    yc = ((y_min + y_max) / 2) / img_h
                    bw = (x_max - x_min) / img_w
                    bh = (y_max - y_min) / img_h
                    f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            if boxes:
                mark_tagged(image_name)       # update cache
                print(f"Saved! Updated tags for {image_name}")
            else:
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                mark_untagged(image_name)     # update cache
                print(f"No boxes — deleted tag file for {image_name}")
            state['result'] = 'next'

        elif key == ord('d'):
            if boxes:
                boxes.pop()
                refresh_screen()
                print("Deleted one box.")
            else:
                print("No boxes to delete.")

        elif key == ord('c'):
            boxes = []
            refresh_screen()
            print("Cleared all boxes.")

        elif key == ord('s'):
            print(f"Skipped {image_name}.")
            state['result'] = 'skip'

        elif key == ord('q'):
            state['result'] = 'quit'

        # Mouse jump result set inside callback
        if isinstance(state['result'], int):
            i = state['result']
            break   # skip the i += 1 below

    else:
        # result is a string ('next' | 'skip' | 'quit')
        if state['result'] == 'quit':
            break
        i += 1
        continue

    # Jumped via dropdown — loop back without incrementing
    if state['result'] == 'quit':
        break

cv2.destroyAllWindows()
print("Editing complete!")