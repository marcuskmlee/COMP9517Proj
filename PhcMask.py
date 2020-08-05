import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv

from PIL import Image
import argparse

from utilis import *
import stretch

parser = argparse.ArgumentParser(description='Meanshift implementation')
parser.add_argument('file', type=str, nargs=1, 
    help='Path to file')

args = parser.parse_args()
filename = args.file[0]

def backgroundSubtraction(filename):
    size = (30, 30)

    elem_type = cv.MORPH_ELLIPSE

    image = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    kernel = cv.getStructuringElement(elem_type, size)

    tophat = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)

    kernel1 = np.ones((5,5),np.uint8)
    O = cv.morphologyEx(tophat, cv.MORPH_OPEN, kernel1)

    return O

img = backgroundSubtraction(filename)

mask = otsuThreshold(img)

path, name = pathname(filename)

cv.imwrite(path+"mask-"+name, mask)

# cv.imshow("bgSub", img)
h = hMaxima(img)
# print("h = "+str(h))
hmax = nFoldDilation(img,h)
cv.imshow("Dilated", hmax)

# print(h)
cv.waitKey()

