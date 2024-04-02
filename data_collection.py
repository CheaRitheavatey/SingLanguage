import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time


cap = cv2.VideoCapture(0) # 0 is the id number for web cam
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

# save image when press button
folder =  "Data/Yes"
counter = 0 # count the number of saved img
while True:
    success, img = cap.read()
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
            imgWhite[0:imageResizeShape[0], wGap:wCal + wGap] = imgResize
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


        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(2)

    # if press the s key it will save the img
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print(counter)