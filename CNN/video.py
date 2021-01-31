from imutils.video import VideoStream
# import the necessary packages
from model_CNN import face_mask_detection_Model
import numpy as np
import argparse
import cv2
import os
from os.path import dirname, join
import time
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
#prototxtPath = os.path.sep.join( "deploy.prototxt")
protoPath = join(dirname(__file__), "deploy.prototxt")
#weightsPath = os.path.sep.join([args["face"],
	#"res10_300x300_ssd_iter_140000.caffemodel"])
weightsPath=join(dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNet(protoPath, weightsPath)
# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = face_mask_detection_Model("model.json", "model_weights.h5")

# load the input image from disk, clone it, and grab the image spatial
# dimensions
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# construct a blob from the image

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
	(104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    #print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    for i in range(0, detections.shape[2]):
    	# extract the confidence (i.e., probability) associated with
    	# the detection
    	confidence = detections[0, 0, i, 2]
    	# filter out weak detections by ensuring the confidence is
    	# greater than the minimum confidence
    	if confidence > args["confidence"]:
    		# compute the (x, y)-coordinates of the bounding box for
    		# the object
    		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    		(startX, startY, endX, endY) = box.astype("int")
    		# ensure the bounding boxes fall within the dimensions of
    		# the frame
    		(startX, startY) = (max(0, startX), max(0, startY))
    		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            # extract the face ROI, convert it from BGR to RGB channel
    		# ordering, resize it to 224x224, and preprocess it
    		face = frame[startY:endY, startX:endX]
    		face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    		face = cv2.resize(face, (100, 100))
    		#face = img_to_array(face)
    		#face = preprocess_input(face)
    		normalized = face / 255.0
    		reshaped = np.reshape(normalized,(-1,100,100,1))
    		reshaped = np.vstack([reshaped])
    		face = np.expand_dims(reshaped, axis=-1)
    		# pass the face through the model to determine if the face
    		# has a mask or not
    		#pred= model.predict_class(face)
    		mask, withoutMask= model.predict_class(face)[0]
    		#print(mask, withoutMask)
            # determine the class label and color we'll use to draw
    		# the bounding box and text
    		label = "Mask" if mask > withoutMask else "No Mask"
    		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    		# include the probability in the label
    		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
    		# display the label and bounding box rectangle on the output
    		# frame
    		cv2.putText(frame, label, (startX, startY - 10),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    # show the output image
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()