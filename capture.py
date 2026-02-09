# import cv2

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# chair_box =[{"box":(66,147,251,298),"status":False},
#             {"box":(433,146,582,344),"status":False},
#             {"box":(7,204,189,403),"status":False},
#             {"box":(418,246,620,394),"status":False}]

# while True:
#     ret,frame = cap.read()
    
#     frame = cv2.flip(frame,1)
    
#     for i in range(len(chair_box)):
#         minX,minY,maxX,maxY = chair_box[i]["box"]
#         cv2.imshow("chair: "+str(int(i)+1),frame[minY:maxY,minX:maxX])
    
#     if (cv2.waitKey(10) & 0xFF == ord("s")):
#         cv2.imwrite("saved_frame.jpg", frame)
#     if (cv2.waitKey(10) & 0xFF == ord("q")):
#         break
    
# cap.release()
# cv2.destroyAllWindows()


