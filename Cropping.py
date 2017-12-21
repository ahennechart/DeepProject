'''
@author: T.Gosset & A.Hennehart

Helped by A.Kacem

'''

import sys
import numpy as np
import cv2

def faceCropping(imgpath):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    gray = cv2.imread(imgpath,0)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
    return (roi_gray)     