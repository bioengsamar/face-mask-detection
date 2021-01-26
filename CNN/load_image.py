from model_CNN import face_mask_detection_Model
import numpy as np
import cv2


model = face_mask_detection_Model("model.json", "model_weights.h5")

font = cv2.FONT_HERSHEY_SIMPLEX
image = cv2.imread('image/ghada.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.imread('image/ghada.jpg')
face_cascade = cv2.CascadeClassifier('C:\\Users\\ht\\Downloads\\opencv-master\\opencv\\data\\haarcascades\\haarcascade_frontalface_default.xml')
#roi = cv2.resize(image, (100, 100))
faces = face_cascade.detectMultiScale(image, 1.3, 5)
for (x, y, w, h) in faces:
            #fc = image[y:y+h, x:x+w]
            roi = cv2.resize(image, (100, 100))
            #X = roi.reshape(-1, 100, 100, 1)
            normalized = roi / 255.0
            reshaped = np.reshape(normalized,(-1,100,100,1))
            reshaped = np.vstack([reshaped])
            pred = model.predict_class(reshaped)

            cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
            image=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            cv2.imwrite('output/output11.png', image)