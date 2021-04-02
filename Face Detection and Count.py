#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2 as cv
import numpy as np


# In[3]:


img = cv.imread('Rudra.jpg')

cv.waitKey(0)
cv.destroyAllWindows()


# In[31]:


#Image, video and live video
def rescaleFrame(frame, scale=0.60):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)


# In[32]:


def rotate(image, angle, rotationPoint=None):
    (height,width) = image.shape[:2]
    if rotationPoint is None:
        rotationPoint = (width//2, height//2)
        
    rotMat = cv.getRotationMatrix2D(rotationPoint, angle, 1.0)
    dimensions = (width, height)
    
    return cv.warpAffine(image, rotMat, dimensions)


# In[8]:


def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)


# In[9]:


#Live video
def changeResolution(width, height):
    vid.set(3, width)
    vid.set(4, height)


# In[40]:


vid = cv.VideoCapture(0, cv.CAP_DSHOW)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    
    isTrue, frame = vid.read()
    frame = cv.flip(frame,1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    faceCount = len(faces)
    
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    text = 'Number of Faces: ' + str(faceCount)
    textsize = cv.getTextSize(text, cv.FONT_HERSHEY_TRIPLEX, 1, 2)[0]
    textX = int((frame.shape[1] - textsize[0])/2)
    textY = (frame.shape[0] - textsize[1])
    
    cv.putText(frame, text, (textX, textY), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 0), 2)
    cv.imshow('Face', frame)
    
    if cv.waitKey(20) & 0xFF == ord('d'):
        break;
        
vid.release()
cv.destroyAllWindows()


# In[ ]:





# In[ ]:




