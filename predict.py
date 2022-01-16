import cv2
import os
from imutils.video import VideoStream
from imutils import face_utils
from tensorflow.keras.models import load_model
import numpy as np
import time
import pygame
import sys
import soundfile as sf
import sounddevice as sd

buzzState = False
print("[INFO] Loading Audio File ....")
data, fs = sf.read('alarm.wav', dtype='float32')  
print("[SUCCESS] Audio File Loaded Successfully ....")

print("[INFO] Loading Haarcascade File...")
face = cv2.CascadeClassifier('haarcascadefiles/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haarcascadefiles/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haarcascadefiles/haarcascade_righteye_2splits.xml')
print("[SUCCESS] Haarcascade File loaded Successfully...")

print("[INFO] Loading Model File...")
model = load_model('modelFile.h5')
print("[SUCCESS] Model File Loaded Successfully...")

path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99,99]
lpred=[99,99]
eyeopen=0
index=0
noface=0

print("[INFO] Predicting...")
while(True):    
    for skip in range(0,5):
        cap.read()
    
    ret, frame = cap.read()
    height,width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (500,height) , (0,0,0) , thickness=cv2.FILLED)
    
    if(len(faces)==0 or len(left_eye)==0 or len(right_eye)==0):
        cv2.putText(frame,"No Face Detected",(10,height-20), font,1,(255,255,255),1,cv2.LINE_AA)
        noface+=1
        if(noface>10 and buzzState==False):
            buzzState = not buzzState
            sd.play(data, fs)
    else:
        noface=0
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

        for (x,y,w,h) in right_eye:
            r_eye=frame[y:y+h,x:x+w]
            count=count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = model.predict(r_eye)[0]
            break

        for (x,y,w,h) in left_eye:
            l_eye=frame[y:y+h,x:x+w]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = model.predict(l_eye)[0]
            break

        if(rpred[0]>rpred[1] and lpred[0]>lpred[1]):
            score=score+1
            eyeopen=0
            cv2.putText(frame,"Closed",(10,height-20), font,1,(255,255,255),2,cv2.LINE_AA)
        else:
            score=score-1
            eyeopen+=1
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
    if(score<0 or eyeopen>2):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(250,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    if(score>5):
        try:            
            buzzState = not buzzState
            if(buzzState):
                sd.play(data, fs)
            else:
                st.stop()
            index+=1
        except:
            pass
        if(thicc<10):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
    else:
        if buzzState and noface<=10:
            buzzState = not buzzState
            sd.stop()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()