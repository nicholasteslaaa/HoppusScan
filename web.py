import cv2
import time
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from detection import workspace_detection

AI = workspace_detection()

app = Flask(__name__)
CORS(app)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Increased resolution for better crops
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ROI = [] # Stores coordinate tuples: (minX, minY, maxX, maxY)
ROI_FRAME = {} # Changed to dictionary to prevent index shifting issues

def generate_frame():
    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Process ROIs
        # We use a temporary dict to avoid flickering during the loop
        new_roi_frames = {}
        for idx, (minX, minY, maxX, maxY) in enumerate(ROI):
            # Ensure coordinates are within frame boundaries to prevent crash
            x1, y1 = max(0, minX), max(0, minY)
            x2, y2 = min(w, maxX), min(h, maxY)
            
            if x2 > x1 and y2 > y1:
                cropped = frame[y1:y2, x1:x2].copy()
                new_roi_frames[idx] = cropped
        
        # Update the global dict once
        global ROI_FRAME
        ROI_FRAME = new_roi_frames

        # Stats for Main Feed
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"ROIs Active: {len(ROI)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        frame = AI.detect(frame)["frame"]
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_roi_stream(idx):
    """Generator for specific ROI streams"""
    while True:
        frame = ROI_FRAME.get(idx)
        if frame is not None:
            frame = AI.detect(frame)["frame"]
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            # If ROI doesn't exist yet, send a black frame or wait
            time.sleep(0.1)
        time.sleep(0.03) # Limit ROI stream to ~30 FPS

@app.route("/get_ROI", methods=["GET"])
def get_ROI():
    chair_idx = request.args.get('chair_idx', type=int)
    if chair_idx is not None:
        return Response(generate_roi_stream(chair_idx),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Missing chair_idx", 400

@app.route("/add_ROI", methods=["POST"])
def add_ROI():
    data = request.get_json()
    try:
        coords = tuple(map(int, data["ROI"].strip().split(" ")))
        ROI.append(coords)
        return jsonify({"status": "success", "index": len(ROI) - 1})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/pop_ROI", methods=["POST"])
def pop_ROI():
    data = request.get_json()
    # Simple logic: your frontend usually sends the index
    idx = data.get("index")
    if idx is not None and 0 <= idx < len(ROI):
        ROI.pop(idx)
        return jsonify({"status": "removed"})
    return jsonify({"error": "invalid index"}), 400

@app.route('/cam_feed')
def video_feed():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # threaded=True is vital for handling multiple streams
    app.run(host='0.0.0.0', port=8000, threaded=True)