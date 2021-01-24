#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 18:03:48 2021

@author: youssef
"""


import cv2
#import time


cam = cv2.VideoCapture(0)

cv2.namedWindow("test")


while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
    img_name = "opencv_frame_{}.png"
    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))


    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    
  
        

cam.release()

cv2.destroyAllWindows()


