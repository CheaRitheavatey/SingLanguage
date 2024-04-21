import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

# plan continue:
# 6. import the classifier
# 7. get prediction and index
# 8. get the output

cap = cv2.VideoCapture(0) # 0 is the id number for web cam
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt") # 6

offset = 20
imgSize = 300

# save image when press button
folder =  "Data/Yes"
counter = 0 # count the number of saved img

labels = ["A", "B", "C", "Yes", "No", "I Love You", "Thank You"]
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3), np.uint8) * 255

       #crop the image
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imageCropShape = imgCrop.shape # contain 3 values: height, width, channel
        
        # calculate to make the image fit inside the white space

        # if the height is greater than width, stretch the height to 300 and calculate the width value
        # if the width is greater than height, stretch the width to 300 and calculate the height value
        aspectRatio = h / w
        # for height
        if aspectRatio > 1:
            k = imgSize / h # stretch the height
            wCal = math.ceil(k * w) # calculate width

            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imageResizeShape = imgResize.shape

            # make the img center in the white
            wGap = math.ceil((imgSize - wCal) / 2)
            # put image white into image crop
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False) # 7 
            # print(prediction, index)
            
        # for width
        else:
            k = imgSize / w # stretch the width
            hCal = math.ceil(k * h) # calculate width

            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imageResizeShape = imgResize.shape

            # make the img center in the white
            hGap = math.ceil((imgSize - hCal) / 2)
            # put image white into image crop
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False) # 7
        
        # 8 green color and text using the labels
        cv2.rectangle(imgOutput, (x - offset,y - offset - 26), (x - offset + 100, y - offset), (0,255,0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x,y-30), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        cv2.rectangle(imgOutput, (x - offset,y - offset), (x + w + offset, y + h + offset), (0,255,0),4)

        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(2)

   