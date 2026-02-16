import cv2
import time
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from database_manager import db_manager
from detection import workspace_detection,CameraStream
import signal
import sys
import numpy as np

app = Flask(__name__)
CORS(app)

db = db_manager()
ROI = db.get_all_data_as_dictionary()

cam_manager = CameraStream().start()

# Create a placeholder
AI = None

@app.before_first_request
def load_model():
    global AI
    if AI is None:
        print("Loading AI Model...")
        AI = workspace_detection()
        
def generate_frame():
    prev_time = 0
    global ROI
    while True:
        frame = cam_manager.get_frame()
        
        for idx in range(len(ROI)):
            minX, minY, maxX, maxY = ROI[idx]["bbox"]
            cropped = frame[minY:maxY, minX:maxX].copy()
            ROI[idx]["frame"] = cropped
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        annotated_frame = AI.detect(frame.copy())
        cv2.putText(annotated_frame["frame"], f"FPS: {int(fps)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated_frame["frame"], f"ROIs Active: {len(ROI)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated_frame["frame"], f"Person Detected: {annotated_frame['people']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def pad_to_size(img, target_w, target_h):
    """Adds black padding to center the original frame without resizing."""
    h, w = img.shape[:2]
    top = (target_h - h) // 2
    bottom = target_h - h - top
    left = (target_w - w) // 2
    right = target_w - w - left
    # Border types: BORDER_CONSTANT uses the [0,0,0] black color
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def generate_roi_stream():
    global ROI
    
    COLS = 4
    last_time = time.perf_counter()
    presence_accumulator = 0.0
    
    max_w, max_h = 0, 0
    while True:
        curr_time = time.perf_counter()
        dt = curr_time - last_time
        last_time = curr_time

        rois = []

        for item in ROI:
            if item["frame"] is not None and item["frame"].size > 0:
                scanned_frame = AI.detect(item["frame"])
                if (scanned_frame["people"] > 0):
                    presence_accumulator += dt
                    if (presence_accumulator >= 1.0):
                        item["timer"] += dt
                        presence_accumulator -= 1.0 
                print(presence_accumulator)
                cv2.putText(scanned_frame["frame"], f"Timr: {int(item['timer'])}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                rois.append(scanned_frame["frame"])
                # Track largest dimensions to create a uniform grid
                max_h = max(max_h, item["frame"].shape[0])
                max_w = max(max_w, item["frame"].shape[1])

        if rois:
            grid_rows = []
            for i in range(0, len(rois), COLS):
                chunk = rois[i : i + COLS]
                padded_chunk = [pad_to_size(c, max_w, max_h) for c in chunk]
                while len(padded_chunk) < COLS:
                    padded_chunk.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))
                grid_rows.append(cv2.hconcat(padded_chunk))

            final_grid = cv2.vconcat(grid_rows)
            
            ret, buffer = cv2.imencode('.jpg', final_grid, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if ret:
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)

        
@app.route("/get_ROI", methods=["GET"])
def get_ROI():
    return Response(generate_roi_stream(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/get_ROI_timer", methods=["GET"])
def get_ROI_timer():
    chair_idx = request.args.get('chair_idx', type=int)
    if chair_idx is not None:
        return jsonify({"timer":"%.2f"%ROI[chair_idx]["timer"]})
    return "Missing chair_idx", 400

@app.route("/add_ROI", methods=["POST"])
def add_ROI():
    data = request.get_json()
    try:
        coords = tuple(map(int, data["ROI"].strip().split(" ")))
        ROI.append({"bbox" : coords,"frame":None, "timer":0})
        db.insert_data(coords,0)
        return jsonify({"status": "success", "index": len(ROI) - 1})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/pop_ROI", methods=["POST"])
def pop_ROI():
    data = request.get_json()
    bbox_str = data["ROI"]
    
    global ROI
    for i in range(len(ROI)):
        x1, y1, x2, y2 = ROI[i]["bbox"]
        current_str = f"{x1} {y1} {x2} {y2}"
        if bbox_str == current_str:
            db.pop_data(ROI[i]["bbox"])
            ROI.pop(i)
            return jsonify({"status": "success"})
            
    return jsonify({"status": "not found"}), 404

@app.route("/list_ROIs", methods=["GET"])
def list_ROIs():
    return jsonify({"rois": [f"{r['bbox'][0]} {r['bbox'][1]} {r['bbox'][2]} {r['bbox'][3]}" for r in ROI]})

@app.route('/cam_feed')
def video_feed():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


def handle_exit(sig, frame):
    print(f"\nSignal {sig} received. Cleaning up...")
    try:
        cam_manager.stop() # Ensure this method calls self.cap.release()
    except:
        pass
    sys.exit(0)

# Register handlers for Ctrl+C (SIGINT) and Ctrl+Z (SIGTSTP)
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTSTP, handle_exit)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, threaded=True,use_reloader=False)