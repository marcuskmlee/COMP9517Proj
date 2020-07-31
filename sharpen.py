import cv2 as cv
import numpy as np
import argparse
from utilis import pathname

parser = argparse.ArgumentParser(description='Constrast stretching')
parser.add_argument('file', type=str, nargs=1, 
    help='Path to file')

args = parser.parse_args()
filename = args.file[0]

def sharpen(filename):
    _a = 1.25
    _sigma = 1

    I = cv.imread(filename)

    L = cv.GaussianBlur(I, (5,5), 0)

    H = cv.subtract(I, L) * _a

    O = cv.add(I, H, dtype=8)

    return O

O = sharpen(filename)

path, filename = pathname(filename)

cv.imwrite(path+'sharpened-'+filename, O)