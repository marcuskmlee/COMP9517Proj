import cv2 as cv
import numpy as np 
import argparse
from utilis import show_image, plot_two

def minFilter(src, dest, radius):    
    
    src = cv.copyMakeBorder(src, radius, radius, radius, radius, 
        cv.BORDER_CONSTANT, value=255)

    for r in range(rows):
        for c in range(cols):
            lowest = 255
            for i in range(-radius, radius):
                for j in range(-radius, radius):
                    val = src[radius + r + i, radius + c + j]
                    if val<lowest:
                        lowest = val
            dest[r,c] = lowest

def maxFilter(src, dest, radius):

    src = cv.copyMakeBorder(src, radius, radius, radius, radius, 
        cv.BORDER_CONSTANT, value=0)

    for r in range(rows):
        for c in range(cols):
            highest = 0
            for i in range(-radius, radius):
                for j in range(-radius, radius):
                    val = src[radius + r + i, radius + c + j]
                    if val>highest:
                        highest = val
            dest[r,c] = highest

parser = argparse.ArgumentParser(description='Min/Max filtering')
parser.add_argument('size', type=int, nargs=1, 
    help='Size of N x N filter')
parser.add_argument('image', type=str, nargs=1, 
    help='Path to the image')
parser.add_argument('--reverse', metavar='-r', nargs='?', 
    help='Reverse the order of min/max filtering', default=False, const=True)
parser.add_argument('--show', metavar='-s', nargs='?', 
    help='Reverse the order of min/max filtering', default=False, const=True)

args = parser.parse_args()

N = args.size[0]
M = args.reverse
show = args.show
name = args.image[0]

_radius = int(N/2)

image = cv.imread(name, cv.IMREAD_GRAYSCALE)

rows, cols = image.shape[:2]

A = np.zeros((rows, cols), dtype=np.uint8)
B = np.zeros((rows, cols), dtype=np.uint8)

if M:
    maxFilter(image, A, _radius)
else: 
    minFilter(image, A, _radius)

# cv.imwrite(f"{name[:-4]}A.png", A)

if M:
    minFilter(A, B, _radius) 
else: 
    maxFilter(A, B, _radius)

if show:
    plot_two("A vs B", A, "A", B, "B")
else:
    cv.imwrite(name[:-4] + "A.png", A)
    cv.imwrite(name[:-4] + "B.png", B)

O = cv.subtract(B, image) if M else cv.subtract(image, B)

if show:
    plot_two("O vs image", image, "Original", O, "B - image_OG")
else:
    cv.imwrite(name[:-4] + "O.png", ~O if M else O)