import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv

from PIL import Image
import argparse

from utilis import *

def otsuThreshold(img):
    blur = cv.GaussianBlur(img,(5,5),0)
    _, mask = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)

    return mask

def backgroundSubtraction(image):
    size = (20, 20)

    elem_type = cv.MORPH_ELLIPSE
    kernel = cv.getStructuringElement(elem_type, size)
    tophat = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)

    kernel1 = np.ones((5,5),np.uint8)
    O = cv.morphologyEx(tophat, cv.MORPH_OPEN, kernel1)

    return O

# img = backgroundSubtraction(filename)
# toHMax = img.copy()
# mask = otsuThreshold(img)
# contours = cv.findContours(mask,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
# cv.drawContours(mask,contours,-1,(0,255,0),3)

# src = backgroundSubtraction(image)
# mask = otsuThreshold(src)
def preprocess(image):
    src = backgroundSubtraction(image)
    mask = otsuThreshold(src)

    # mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

    return mask, src

# img = backgroundSubtraction(filename)
# toHMax = img.copy()
# mask = otsuThreshold(img)
# # contours = cv.findContours(mask,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
# # cv.drawContours(mask,contours,-1,(0,255,0),3)

# src = backgroundSubtraction(image)
# mask = otsuThreshold(src)

# cv.imwrite(path+"mask-"+name, mask)

# # cv.imshow("bgSub", img)
# h = hMaxima(toHMax)
# # print("h = "+str(h))
# hmax = nFoldDilation(toHMax,h)
# cv.imshow("Dilated", hmax)

# # print(h)
# cv.waitKey()
