import cv2
import time
from ultralytics import YOLO
import threading
import torch 

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
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.frame.copy()
    
    def release_cam(self):
        self.cap.release()

    
class workspace_detection:
    def __init__(self,model_path:str = 'yolov8n.engine') -> None:
        self.model = YOLO(model_path, task="detect")
        self.device = 0 if torch.cuda.is_available() else "cpu"
    
    def detect(self,frame)->dict:
        # Create a copy so we don't draw on the original frame
        draw_frame = frame.copy() 
        
        num_of_people = 0
        for results in self.model.predict(source=frame, stream=True, device=self.device, imgsz=320, verbose=False):
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for cls, box in zip(boxes.cls, boxes.xyxy):
                        if int(cls) == 0: # If person/object class matches
                            # Draw ONLY on the copy
                            num_of_people += 1
                            cv2.rectangle(draw_frame, 
                                        (int(box[0]), int(box[1])), 
                                        (int(box[2]), int(box[3])), 
                                        (0, 255, 255), 2)
        return {"frame":draw_frame,"people":num_of_people}
    
    def get_center_box(self,xy_1:tuple,xy_2:tuple) -> tuple:
        cx = int((xy_1[0] + xy_2[0]) / 2)
        cy = int((xy_1[1] + xy_2[1]) / 2)
        return (cx,cy)

    def is_point_inside(self,cx:int, cy:int, rect:tuple) -> bool:
        return cx >= rect[0] and cx <= rect[2] and cy >= rect[1] and cy <= rect[3]
    