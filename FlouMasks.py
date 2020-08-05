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

img = cv.imread(filename,cv.IMREAD_GRAYSCALE)

img = stretch(img)

mask = otsuThreshold(img)

path, name = pathname(filename)

cv.imwrite(path+"mask-"+name, mask)

h = hMaxima(img)
# print("h = "+str(h))
hmax = nFoldDilation(img,h)
cv.imshow("Dilated", hmax)

# print(h)
cv.waitKey()