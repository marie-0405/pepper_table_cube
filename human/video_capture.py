import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import time

camera = cv2.VideoCapture(0,cv2.CAP_DSHOW) # Camera channel
for i1 in range(0, 20): 
    cap1 = cv2.VideoCapture(i1, cv2.CAP_DSHOW )
    if cap1.isOpened(): 
        print("VideoCapture(", i1, ") : Found")
    else:
        print("VideoCapture(", i1, ") : None")
    cap1.release() 

# Settings of video file
fps = int(camera.get(cv2.CAP_PROP_FPS))                    # FPS
fps = 30
w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))              # 
h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))             
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        
video = cv2.VideoWriter('video.mp4', fourcc, fps, (w, h))

print(w, h)
while True:
    ret, frame = camera.read()    # Get 1 frame        
    cv2.imshow('camera', frame)             # Display frame
    video.write(frame)
 
    # break if q key is entered
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
camera.release()
cv2.destroyAllWindows()