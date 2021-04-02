import numpy as np
import cv2 as cv

def rescaleFrame(frame, scale=0.20):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

people = ['Ben Afflek', 'Elton John','Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Rudra Makwana']

# img = cv.imread(r'C:\Users\rudra\Desktop\Machine Learning\Real-Time Face Recognition\Faces\val\madonna\4.jpg')
#img = rescaleFrame(img)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# for (x, y, w, h) in faces:
#     faces_roi = gray[y:y+h, x:x+w]
#     cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

#     label, confidance = face_recognizer.predict(faces_roi)
#     cv.putText(img, people[label], (x, y), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)

#     print(confidance)
    
# cv.imshow('Face', img)

# cv.waitKey(0)


vid = cv.VideoCapture(0, cv.CAP_DSHOW)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    
    isTrue, frame = vid.read()
    frame = cv.flip(frame,1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)
    
    for (x, y, w, h) in faces:
        faces_roi = gray[y:y+h, x:x+w]
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        label, confidance = face_recognizer.predict(faces_roi)
        cv.putText(frame, people[label], (x, y), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
        cv.putText(frame, "Confidance:"+str(confidance), (x, y+h), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 1)
    
    cv.imshow('Face', frame)
    
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
        
vid.release()
cv.destroyAllWindows()