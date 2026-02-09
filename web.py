import cv2
import time
import threading
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from detection import workspace_detection

AI = workspace_detection()

app = Flask(__name__)
CORS(app)


ROI = [] 
ROI_FRAME = {} 




class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = cv2.flip(frame, 1)
            time.sleep(0.01) # Small delay to yield CPU

    def get_frame(self):
        with self.lock:
            return self.frame.copy()

# Initialize the global camera thread
cam_manager = CameraStream().start()

def generate_frame():
    prev_time = 0
    while True:
        frame = cam_manager.get_frame()


        frame = cv2.flip(frame, 1)
        
        new_roi_frames = {}
        for idx in range(len(ROI)):
            minX, minY, maxX, maxY = ROI[idx]
            cropped = frame[minY:maxY, minX:maxX].copy()
            new_roi_frames[idx] = cropped
        
        global ROI_FRAME
        ROI_FRAME = new_roi_frames

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"ROIs Active: {len(ROI)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # frame = AI.detect(frame)["frame"]
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_roi_stream(idx):
    while True:
        frame = ROI_FRAME.get(idx)
        if frame is not None:
            frame = AI.detect(frame)["frame"]
            ret, buffer = cv2.imencode('.jpg', frame)
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
    bbox = data["ROI"]

    for i in range(len(ROI)):
        x1,y1,x2,y2 = ROI[i]
        if (bbox == f"{x1} {y1} {x2} {y2}"):
            ROI.pop(i)
            ROI_FRAME.pop(i)
        
    
    return jsonify({"status": "success", "index": len(ROI) - 1})

@app.route("/list_ROIs", methods=["GET"])
def list_ROIs():
    # Returns a list of strings "x1 y1 x2 y2"
    return jsonify({"rois": [f"{r[0]} {r[1]} {r[2]} {r[3]}" for r in ROI]})

@app.route('/cam_feed')
def video_feed():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, threaded=True)