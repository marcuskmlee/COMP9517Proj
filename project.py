# import the necessary packages
import cv2
import numpy as np
import math  

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift

from PIL import Image
import math

from cell import Cell

def plot_three_images(figure_title, image1, label1, image2, label2, image3, label3):
    fig = plt.figure()
    fig.suptitle(figure_title)

    # Display the first image
    fig.add_subplot(1, 3, 1)
    plt.imshow(image1)
    plt.axis('off')
    plt.title(label1)
    
    # Display the second image
    fig.add_subplot(1, 3, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.title(label2)
    
    # Display the third image
    fig.add_subplot(1, 3, 3)
    plt.imshow(image3)
    plt.axis('off')
    plt.title(label3)

    plt.show()

def contrast_smoothing(image):
    max = 0
    min = 255
    smooth = image.copy()
    (h, w, d) = smooth.shape
    pixels = [[0 for x in range(w)] for y in range(h)]

    for x in range(w):
        for y in range(h):
            pixelval, q, p = image[y, x]
            intval = pixelval.astype(np.int32)
            pixels[y][x] = intval
            if intval < min:
                min = intval
            if intval > max:
                max = intval
    for x in range(w):
        for y in range(h):
            intval = pixels[y][x]
            newval = (intval-min) * ((255)/(max - min))
            smooth[y, x] = (newval, newval, newval)
    return smooth.astype(np.uint8)

def flood_adjacent(image, y, x, i):
    (h, w, d) = image.shape
    pixelval, q, p = image[y, x]
    if (pixelval == 255):
        image[y, x] = (i, i, i)
        if (y+1 < h):
            flood_adjacent(image, y+1, x, i)
        if (y-1 >= 0):
            flood_adjacent(image, y-1, x, i)
        if (x+1 < w):
            flood_adjacent(image, y, x+1, i)
        if (x-1 >= 0):
            flood_adjacent(image, y, x-1, i)

def flood(image):
    flooded = image.copy()
    (h, w, d) = flooded.shape
    i = 1
    for x in range(w):
        for y in range(h):
            pixelval, q, p = flooded[y, x]
            if (pixelval == 255):
                flood_adjacent(flooded, y, x, i)
                i = i + 1
    return flooded


def process_image(image_path):

    image = cv2.imread(image_path)

    result = top_hat(image, 21)


    thresh = cv2.threshold(result,100,255,cv2.THRESH_BINARY)[1]

    #erode single pixel around each cell to remove small artifacts and separate lightly touching cells
    #use for PhC image set but not for others
    eroded = cv2.erode(thresh, None, iterations=1)

    segmented = flood(eroded)

    plot_three_images(image_path, image, "Original Image", eroded, "Threshold", segmented, "Segmented")

    return segmented

def top_hat(image, N):

    max_filtered = image.copy()
    max_filtered = cv2.erode(max_filtered, None, iterations=15)

    min_filtered = cv2.dilate(max_filtered, None, iterations=15)

    result = image.astype(np.int32) - min_filtered.astype(np.int32)

    final = contrast_smoothing(result)
    

    return final

#get properties of each cell and store are respective index, remove cells below size threshold
def count_cells(image):
    cells = []
    (h, w, d) = image.shape
    for x in range(w):
        for y in range(h):
            pixelval, q, p = image[y, x]
            if (pixelval != 0):
                existing = False
                for cell in cells:
                    if (cell.get_id() == pixelval):
                        existing = True
                        cell.update_bound(x,y)
                if (existing == False):
                    new_cell = Cell(pixelval, x, y)
                    cells.append(new_cell)

    return cells


image_path = "COMP9517 20T2 Group Project Image Sequences/PhC-C2DL-PSC/Sequence 1/t000.tif"

segmented = process_image(image_path)

cells = count_cells(segmented)

for cell in cells:
    print (cell)
