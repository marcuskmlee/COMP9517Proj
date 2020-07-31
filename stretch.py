import cv2 as cv
import numpy as np
import argparse
from utilis import pathname
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Constrast stretching')
parser.add_argument('file', type=str, nargs=1, 
    help='Path to file')

args = parser.parse_args()
filename = args.file[0]

def stretch(filename):

    image = cv.imread(filename)

    minVal = np.amin(image)
    maxVal = np.amax(image)

    mod = 255.0/(maxVal-minVal)

    table = np.array( [(x-minVal)*mod for x in range(256)] )

    draw = cv.LUT(image, table)

    draw = np.uint8(draw)

    return draw

draw = stretch(filename)

path, filename = pathname(filename)

cv.imwrite(path+"stretched-"+filename, draw)