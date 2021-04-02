# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:26:02 2021

@author: rudra
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

#Image, video and live video
def rescaleFrame(frame, scale=0.20):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)
    
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

people = ['Ben Afflek', 'Elton John','Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Rudra Makwana']

dirTest = r'C:\Users\rudra\Desktop\Machine Learning\Real-Time Face Recognition\Faces\train'

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(dirTest, person)
        label = people.index(person)
        
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                faces_roi = gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)
                
create_train()

print("Training Done")

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)



