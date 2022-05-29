import tkinter as tk
import cv2
import csv
import os
import numpy as np
from PIL import Image,ImageTk
import pandas as pd
import datetime
import time

window = tk.Tk()
window.title("Face Recognition Attendance System ")
window.geometry('1280x620')
window.configure()


def take_img():

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(r'C:\Users\BoardHP\Desktop\project\haarcascade_frontalface_default.xml')
    ID = txt.get()
    Name = txt2.get()
    sampleNum = 0
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
            sampleNum = sampleNum + 1
    
            cv2.imwrite(r"C:\Users\BoardHP\Desktop\project\dataset\ " + Name + "." + ID + '.' + str(sampleNum) + ".jpg",
                                gray[y:y + h, x:x + w])
            cv2.imshow('Frame', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif sampleNum > 100:
            break
    cam.release()
    cv2.destroyAllWindows()
        
    res = "Images Saved  : " + ID + " Name : " + Name
    Notification.configure(text=res, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
    Notification.place(x=250, y=400)
      


def trainimg():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    global detector
    detector = cv2.CascadeClassifier(r"C:\Users\BoardHP\Desktop\project\haarcascade_frontalface_default.xml")
    try:
        global faces,Id
        faces, Id = getImagesAndLabels(r"C:\Users\BoardHP\Desktop\project\dataset")
    except Exception as e:
        l='no files is in folder'
        Notification.configure(text=l, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
        Notification.place(x=350, y=400)

    recognizer.train(faces, np.array(Id)) 
    try:
        recognizer.save(r"C:\Users\BoardHP\Desktop\project\model\trained_model2.yml")
    except Exception as e:
        q=' "model" not found'
        Notification.configure(text=q, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
        Notification.place(x=350, y=400)

    res = "Model Trained"  
    Notification.configure(text=res, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
    Notification.place(x=250, y=400)

	

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    
    faceSamples = []
    Ids = [] #list
    
    for imagePath in imagePaths:

        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image

        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)
        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    return faceSamples, Ids


window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


def on_closingfun():
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Do you want to quit the window?"):
        window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closingfun)



Notification = tk.Label(window, text="All things good", bg="Green", fg="white", width=15, height=3)

lbl = tk.Label(window, text="Enter id", width=20, height=2, fg="black", font=('times', 20, 'italic bold '))
lbl.place(x=200, y=200)
	
	
message = tk.Label(window, text="Face Recognition Attendance System ", bg="cyan", fg="black", width=50,
                   height=3, font=('times', 30, 'italic bold '))

message.place(x=80, y=20)	

txt = tk.Entry(window, validate="key", width=20,  fg="red")
txt.place(x=550, y=210)

lbl2 = tk.Label(window, text="Enter Name", width=20, fg="black",  height=2, font=('times', 20, 'italic bold '))
lbl2.place(x=200, y=300)

txt2 = tk.Entry(window, width=20, fg="red")
txt2.place(x=550, y=310)

takeImg = tk.Button(window, text="Take Images",command=take_img,fg="black", bg="green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 20, 'italic bold '))
takeImg.place(x=200, y=500)

trainImg = tk.Button(window, text="Train Images",fg="black",command=trainimg ,bg="green"  ,width=20  ,height=3, activebackground = "Red",font=('times', 20, 'italic bold '))
trainImg.place(x=590, y=500)

window.mainloop()