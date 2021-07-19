import cv2
from time import sleep

fps = 10
while 1:
    try:
        img = cv2.imread('img.jpg')
        cv2.imshow('window', img)
        cv2.waitKey(1)
    except:
        pass

    sleep(1/fps)
