import numpy as np
import cv2
from model_CNN import face_mask_detection_Model
model = face_mask_detection_Model("model.json", "model_weights.h5")


face_cascade = cv2.CascadeClassifier('C:\\Users\\ht\\Downloads\\opencv-master\\opencv\\data\\haarcascades\\haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
#eye_cascade = cv2.CascadeClassifier('C:\\Users\\ht\\Downloads\\opencv-master\\opencv\\data\\haarcascades\haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        fc = gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (100, 100))
        #X = roi.reshape(-1, 100, 100, 1)
        normalized = roi / 255.0
        reshaped = np.reshape(normalized,(-1,100,100,1))
        #reshaped = np.vstack([reshaped])
        pred = model.predict_class(reshaped)
    
        cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
        
            
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()