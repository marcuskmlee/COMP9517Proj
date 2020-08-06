import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv

from PIL import Image
import argparse

from utilis import *

def otsuThreshold(img, start, end):
    blur = cv.GaussianBlur(img,(5,5),0)
    # mask = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C,\
    #         cv.THRESH_BINARY,11,2)

    _, mask = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    return mask

def preprocess_image(img):
    img, minVal, maxVal = stretch(img)
    mask = otsuThreshold(img, maxVal, maxVal)

    print(f"minVal: {minVal}, maxVal: {maxVal}, difference: {maxVal - minVal}")

    return mask, img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='stretching')
    parser.add_argument('file', type=str, nargs=1, help='Path to file')

    args = parser.parse_args()
    filename = args.file[0]

    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    show_image(img, "Original")

    mask, img = preprocess_image(img)

    plot_two("Mask", img, "Original", mask, "Mask")