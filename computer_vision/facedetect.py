'''
Tay Mei Lan @ 27 Mar 2022

Source/References: From OpenCV pre-installed packages scripts
- Detects faces in a moving video

Pre-requisite:
pip install opencv-python

To customise:
- Modify cascade filepath to requirements 
'''

import cv2
import sys

cascPath = 'C:\\Python38-32\\Lib\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'  #sys.argv[1]
#cascPath = 'C:\\Python38-32\\Lib\site-packages\\cv2\\data\\haarcascade_eye_tree_eyeglasses.xml'  #sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
