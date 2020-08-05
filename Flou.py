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

def preprocess_image():
    img = stretch(img)

    mask = otsuThreshold(img)

    return mask