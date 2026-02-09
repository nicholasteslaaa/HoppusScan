import cv2
import time
from ultralytics import YOLO


class canvas:
    def __init__(self) -> None:
        # Global state
        self.annotations = []
        self.is_drawing = False
        self.ix, self.iy = -1, -1
        self.cx, self.cy = -1, -1 # Current mouse position
    
    def mouse_callback(self, event, x, y, flags, param):
        H, W = 480, 640 
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.ix, self.iy = x, y
            self.cx, self.cy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing: # Only update current position if dragging
                self.cx, self.cy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            x1, x2 = sorted([self.ix, x])
            y1, y2 = sorted([self.iy, y])
            
            if (x2 - x1) > 10 and (y2 - y1) > 10: 
                self.annotations.append((x1, y1, x2, y2))

    
class workspace_detection:
    def __init__(self,model_path:str = 'yolov8n.engine',camera_idx:int = 0,frame_width:int = 640,frame_height:int = 480) -> None:
        self.model = YOLO(model_path, task="detect")
        
        # self.cap = cv2.VideoCapture(camera_idx) # 0 for webcam
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        # cv2.namedWindow("Live Annotator")
        
        # self.draw_canvas = canvas()
        # cv2.setMouseCallback("Live Annotator", self.draw_canvas.mouse_callback)
        
        
    
    def detect(self,roi_frame)->dict:
        # Create a copy so we don't draw on the original frame
        draw_frame = roi_frame.copy() 
        
        num_of_people = 0
        # Run prediction on the roi_frame (original)
        for results in self.model.predict(source=roi_frame, stream=True, device=0, imgsz=320, verbose=False):
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
    
    
#     def main(self):
#         prev_time = 0
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
            

#             # 1. Draw persistent saved boxes (Green)
#             annotations = self.draw_canvas.annotations
#             for i in range(len(annotations)):
#                 minx, miny, maxx, maxy = annotations[i]
                
#                 # Grab the ROI
#                 roi_crop = frame[miny:maxy, minx:maxx]
                
#                 # Safety Check: Only proceed if the crop is not empty
#                 if roi_crop.size > 0 and roi_crop.shape[0] > 5 and roi_crop.shape[1] > 5:
#                     ROI = self.detect(roi_crop)
#                     cv2.imshow(f"chair: {i+1}", ROI["frame"])
                    
#                 # Still draw the green box on the main frame
#                 cv2.rectangle(frame, (minx, miny), (maxx, maxy), (0, 255, 0), 2)

#             # 2. Draw the live "active" box (Red) while dragging
#             if self.draw_canvas.is_drawing:
#                 cv2.rectangle(frame, (self.draw_canvas.ix, self.draw_canvas.iy), (self.draw_canvas.cx, self.draw_canvas.cy), (0, 0, 255), 2)

#             curr_time = time.time()
#             fps = 1 / (curr_time - prev_time)
#             prev_time = curr_time
            
#             scanned_frame = self.detect(frame)
#             cv2.imshow("Live Annotator", scanned_frame["frame"])
            
#             fps_text = "FPS: "+str(int(fps))
#             num_of_people = "Person: " + str(int(scanned_frame["people"])) 
#             cv2.putText(frame, fps_text, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
#             cv2.putText(frame, num_of_people, (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            
            
#             key = cv2.waitKey(1) & 0xFF # Low delay for high responsiveness
#             if key == ord('q'):
#                 break
#             elif key == ord('c'):
#                 annotations = []
#             elif key == ord('u'):
#                 if annotations: annotations.pop()


#         self.cap.release()
#         cv2.destroyAllWindows()

#         print("Final Coordinates:")
#         for ann in annotations:
#             print(f"({ann[0]},{ann[1]},{ann[2]},{ann[3]})")\

# # if __name__ == "__main__":
# #     wd = workspace_detection()
# #     wd.main()