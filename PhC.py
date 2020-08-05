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
    size = (30, 30)

    elem_type = cv.MORPH_ELLIPSE

    kernel = cv.getStructuringElement(elem_type, size)

    tophat = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)

    kernel1 = np.ones((5,5),np.uint8)
    O = cv.morphologyEx(tophat, cv.MORPH_OPEN, kernel1)

    return O

def preprocess_image(image):

    src = backgroundSubtraction(image)
    mask = otsuThreshold(src)

    return mask