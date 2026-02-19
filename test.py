import cv2
import numpy as np
from database_manager import db_manager
from detection import workspace_detection
import time

AI = workspace_detection("yolov8n.pt")

db = db_manager()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
COLS = 4

def pad_to_size(img, target_w, target_h):
    """Adds black padding to center the original frame without resizing."""
    h, w = img.shape[:2]
    top = (target_h - h) // 2
    bottom = target_h - h - top
    left = (target_w - w) // 2
    right = target_w - w - left
    # Border types: BORDER_CONSTANT uses the [0,0,0] black color
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

last_time = time.perf_counter()
presence_accumulator = 0.0

all_items = db.get_all_data_as_dictionary()
max_w, max_h = 0, 0
while True:
    curr_time = time.perf_counter()
    dt = curr_time - last_time
    last_time = curr_time
        
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    notated_fram = frame.copy()
    if not ret: break

    raw_crops = []

    for item in all_items:
        x_min, y_min, x_max, y_max = item["bbox"]
        cv2.rectangle(notated_fram, 
                    (x_min, y_min), 
                    (x_max, y_max), 
                    (0, 255, 255), 2)
        crop = frame[y_min:y_max, x_min:x_max].copy()
        if crop.size > 0:
            scanned_frame = AI.detect(crop)
            if (scanned_frame["people"] > 0):
                presence_accumulator += dt
                if (presence_accumulator >= 1.0):
                    item["timer"] += dt
                    presence_accumulator -= 1.0 
            print(presence_accumulator)
            cv2.putText(scanned_frame["frame"], f"Timr: {int(item['timer'])}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            raw_crops.append(scanned_frame["frame"])
            # Track largest dimensions to create a uniform grid
            max_h = max(max_h, crop.shape[0])
            max_w = max(max_w, crop.shape[1])

    if raw_crops:
        grid_rows = []
        for i in range(0, len(raw_crops), COLS):
            chunk = raw_crops[i : i + COLS]
            padded_chunk = [pad_to_size(c, max_w, max_h) for c in chunk]
            while len(padded_chunk) < COLS:
                padded_chunk.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))
            grid_rows.append(cv2.hconcat(padded_chunk))

        final_grid = cv2.vconcat(grid_rows)
        cv2.imshow("HoppusScan Original Scale Grid", final_grid)

    cv2.imshow("Live Feed", notated_fram)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()