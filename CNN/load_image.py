from model_CNN import face_mask_detection_Model
import numpy as np
import cv2


model = face_mask_detection_Model("model.json", "model_weights.h5")

font = cv2.FONT_HERSHEY_SIMPLEX
image = cv2.imread('image/44.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.imread('image/44.jpg')
face_cascade = cv2.CascadeClassifier('C:\\Users\\ht\\Downloads\\opencv-master\\opencv\\data\\haarcascades\\haarcascade_frontalface_default.xml')
#roi = cv2.resize(image, (100, 100))
faces = face_cascade.detectMultiScale(image, 1.3, 5)
for (x, y, w, h) in faces:
            
            roi = cv2.resize(image, (100, 100))
            pred = model.predict_class(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
            image=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            cv2.imwrite('output/output6.png', image)