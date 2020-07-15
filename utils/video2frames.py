import cv2
import os 

video_path = os.path.expanduser("~/Downloads/output1_compressed.avi")
out_path = os.path.expanduser("~/Downloads/out_video_compressed/")

cap = cv2.VideoCapture(video_path)
frame_name = "Udacity_Dataset1_frame_"

i = 0
while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    name = frame_name + str(i) + ".jpg"
    cv2.imwrite(out_path + name,frame,[cv2.IMWRITE_JPEG_QUALITY, 0])
    i += 1
