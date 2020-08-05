import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv

from PIL import Image
import argparse

from utilis import *

parser = argparse.ArgumentParser(description='Meanshift implementation')
parser.add_argument('file', type=str, nargs=1, 
    help='Path to file')

args = parser.parse_args()
filename = args.file[0]

img = cv.imread(filename)

img = stretch(img)

mask = otsuThreshold(img)

path, name = pathname(filename)

cv.imwrite(path+"mask-"+name, mask)