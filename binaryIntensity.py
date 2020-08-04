import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv

from scipy import ndimage as ndi
from skimage import color
from skimage.future import graph
from sklearn.cluster import MeanShift, estimate_bandwidth

from skimage.segmentation import watershed, quickshift, mark_boundaries, slic
from skimage.feature import peak_local_max
from skimage.util import img_as_float

from PIL import Image
import argparse

from utilis import *
from utilis import _meanshift

parser = argparse.ArgumentParser(description='Meanshift implementation')
parser.add_argument('file', type=str, nargs=1, 
    help='Path to file')

args = parser.parse_args()
filename = args.file[0]

img = cv.imread(filename, cv.IMREAD_ANYDEPTH)

# show_image(img, "Original")

def otsuThreshold(img):
    blur = cv.GaussianBlur(img,(5,5),0)
    _, mask = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)

    return mask

# mask = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv.THRESH_BINARY, 11, 2)

# img2 = Image.open(filename)
# size = (500,500)
# img2.thumbnail(size)

# img2 = np.array(img2)[:, :, :3]

# ms_labels, ms_out = _meanshift(img2)

# mappings = aggregate(ms_labels)
# cluster = mappings[0]

# def _find(cluster, ms_labels):
#     rows, cols = ms_labels.shape

#     for x in range(rows):
#         for y in range(cols):
#             if ms_labels[x, y] == cluster:
#                 return x, y
    
#     return 0, 0

# x, y = _find(cluster, ms_labels)

# print(ms_out[x, y])
# exit(1)

# plot_two("meanshift", img, "OG", ms_out, "ms")

# mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

_, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

plot_two("meanshift", img, "OG", mask, "mask")

colour = (255, 255, 255)

for i in range(len(contours)):
    cv.drawContours(img, contours, i, colour, 2)

show_image(img, "Contours")

path, name = pathname(filename)
cv.imwrite(path + "mask-" + name, mask)
