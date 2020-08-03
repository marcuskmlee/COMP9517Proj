import cv2 as cv
import numpy as np 
import argparse
from utilis import show_image, plot_two, pathname

parser = argparse.ArgumentParser(description='Min/Max filtering')
parser.add_argument('image', type=str, nargs=1, 
    help='Path to the image')
parser.add_argument('size', type=int, nargs="?", 
    help='Size of N x N filter', default=30)
parser.add_argument('--show', metavar='-s', nargs='?', 
    help='Reverse the order of min/max filtering', default=False, const=True)
parser.add_argument('--kernel_type', nargs=1, type=str,  
    help='Kernel Structure', default="ellipse")

args = parser.parse_args()

N = args.size
elem_type = args.kernel_type[0]
show = args.show
name = args.image[0]

size = (N, N)

elem_type_dict = {
    "rect"      : cv.MORPH_RECT,
    "ellipse"   : cv.MORPH_ELLIPSE,
    "cross"     : cv.MORPH_CROSS
}

if elem_type in elem_type_dict:
    elem_type = elem_type_dict[elem_type]
else:
    elem_type = cv.MORPH_RECT

image = cv.imread(name, cv.IMREAD_GRAYSCALE)

kernel = cv.getStructuringElement(elem_type, size)

tophat = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)

kernel1 = np.ones((5,5),np.uint8)
O = cv.morphologyEx(tophat, cv.MORPH_OPEN, kernel1)

if show:
    plot_two("O vs image", image, "Original", O, "TOPHAT")
else:
    path, name = pathname(name)
    cv.imwrite(path + "O-" + name, O)