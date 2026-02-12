import cv2
import time
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from database_manager import db_manager
from detection import workspace_detection,CameraStream
import signal
import sys

AI = workspace_detection()

app = Flask(__name__)
CORS(app)

db = db_manager()
ROI = db.get_all_data_as_dictionary()

cam_manager = CameraStream().start()
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
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"ROIs Active: {len(ROI)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_roi_stream(idx):
    global ROI
    
    curr_time = time.time()
    while True:
        if (len(ROI) > idx):
            frame = ROI[idx]["frame"]
            if frame is not None:
                detection = AI.detect(frame)
                frame = detection["frame"]
                if (detection["people"] > 0):
                    new_time = time.time()
                    delta_time = new_time - curr_time
                    curr_time = new_time
                    
                    ROI[idx]["timer"] += delta_time
                    db.update_timer(ROI[idx]["bbox"],ROI[idx]["timer"])
                if frame.shape[0] > 0 and frame.shape[1] > 0:
                    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    if ret:
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.1)
        time.sleep(0.03)
        
@app.route("/get_ROI", methods=["GET"])
def get_ROI():
    chair_idx = request.args.get('chair_idx', type=int)
    if chair_idx is not None:
        return Response(generate_roi_stream(chair_idx),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Missing chair_idx", 400

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
