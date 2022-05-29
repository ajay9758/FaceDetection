import os
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r'C:\Users\BoardHP\Desktop\project\model\trained_model2.yml')
detector = cv2.CascadeClassifier(r"C:\Users\BoardHP\Desktop\project\haarcascade_frontalface_default.xml")

faceCascade = cv2.CascadeClassifier(r'C:\Users\BoardHP\Desktop\project\haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)

def attendence(id):
    with open(r'C:\Users\BoardHP\Desktop\project\Attendence.csv','a+') as f:
        myData = f.readline()
        idlist = []
        for line in myData:
            entry = line.split(',')
            idlist.append(entry[0])
        if id not in idlist:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'{id},{tStr},{dStr}')

while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    faces=faceCascade.detectMultiScale(gray, 1.2,4)
    
    for(x,y,w,h) in faces:
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 7)
        cv2.putText(im, str(Id), (x,y-40),font, 2, (255,255,255), 3)
        
    cv2.imshow('image',im)
        
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

attendence(Id)
cam.release()
cv2.destroyAllWindows()