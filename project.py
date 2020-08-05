# import the necessary packages
import cv2
import numpy as np
import math 
import glob 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import ndimage as ndi
from scipy.spatial import distance
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift

from PIL import Image
import math

from cell import Cell

#not currently used, only for testing
def plot_two_images(figure_title, image1, label1, image2, label2):
    fig = plt.figure()
    fig.suptitle(figure_title)

    # Display the first image
    fig.add_subplot(1, 2, 1)
    plt.imshow(image1)
    plt.axis('off')
    plt.title(label1)
    
    # Display the second image
    fig.add_subplot(1, 2, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.title(label2)
    

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



def process_image(image):

    result = top_hat(image, 21)

    thresh = cv2.threshold(result,100,255,cv2.THRESH_BINARY)[1]

    #erode single pixel around each cell to remove small artifacts and separate lightly touching cells
    #use for PhC image set but not for others
    #eroded = cv2.erode(thresh, None, iterations=1)

    return thresh

def top_hat(image, N):

    max_filtered = image.copy()
    max_filtered = cv2.erode(max_filtered, None, iterations=15)

    min_filtered = cv2.dilate(max_filtered, None, iterations=15)

    result = image.astype(np.int32) - min_filtered.astype(np.int32)

    final = contrast_smoothing(result)
    

    return final

#recursive flooding of cell
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

#labels each cell with different value
def flood(image):
    flooded = image.copy()
    flooded = flooded.astype(np.int32)
    (h, w, d) = flooded.shape
    i = 1
    for x in range(w):
        for y in range(h):
            pixelval, q, p = flooded[y, x]
            if (pixelval == 255):
                flood_adjacent(flooded, y, x, i)
                i = i + 1
                #skips i = 255 to avoid infinite recursion
                if (i == 255):
                    i = i + 1
    return flooded

#get properties of each cell and store are respective index, remove cells below size threshold
# def count_cells(image):
#     cells = []
#     (h, w, d) = image.shape
#     for x in range(w):
#         for y in range(h):
#             pixelval, q, p = image[y, x]
#             if (pixelval != 0):
#                 existing = False
#                 for cell in cells:
#                     if (cell.get_id() == pixelval):
#                         existing = True
#                         cell.update_bound(x,y)
#                 if (existing == False):
#                     print(pixelval)
#                     print(x)
#                     print(y)
#                     new_cell = Cell(pixelval, x , y)
#                     cells.append(new_cell)

#     return cells

def draw_bounding_box(image, cells):
    drawn = image.copy()
    for cell in cells:
        #red bounding box for cells
        colour = (0, 0, 255)
        if (cell.get_dividing()):
            #blue bounding box when cell division
            colour = (255, 0, 0)

        drawn = cv2.rectangle(drawn, (cell.get_x_min(), cell.get_y_min()), (cell.get_x_max(), cell.get_y_max()), colour, 1)
    
    return drawn

def count_cell_divisions(cells):
    count = 0
    for cell in cells:
        if (cell.get_dividing()):
            count = count + 1
    return count


def in_image(image_no, cell_id):
    if (image_no < 0 or image_no >= len(sequence)):
        return False
    for cell in sequence[image_no]:
        if (cell.get_id() == cell_id):
            return True
    return False

def get_cell(image_no, cell_id):
    for cell in sequence[image_no]:
        if (cell.get_id() == cell_id):
            return cell
    return None

def calculate_speed(cell_id):
    if (in_image(cur_image - 1, cell_id)):
        return distance.euclidean(get_cell(i, cell_id).get_centre(), get_cell(cur_image, cell_id).get_centre())
    return 0

def calculate_total_distance(cell_id):
    total = 0
    for i in range(cur_image - 1):
        if (in_image(i, cell_id)):
            total = total + distance.euclidean(get_cell(i, cell_id).get_centre(), get_cell(i + 1, cell_id).get_centre())
    return total

def calculate_net_distance(cell_id):
    for i in range(cur_image):
        if (in_image(i, cell_id)):
            return distance.euclidean(get_cell(i, cell_id).get_centre(), get_cell(cur_image, cell_id).get_centre())
    return 0

def show_cell_details(x, y):
    for cell in sequence[cur_image]:
        cell_id = cell.get_id()
        if (cell.contains(x, y) and in_image(cur_image, cell_id)):
            print("Speed: " + str(calculate_speed(cell_id)))
            total_distance = calculate_total_distance(cell_id)
            print("Total Distance: " + str(total_distance))
            net_distance = calculate_net_distance(cell_id)
            print("Net Distance: " + str(net_distance))
            confinement = 0
            if (net_distance != 0):
                confinement = (total_distance / net_distance)
            print("Confinement Ratio: " + str(confinement))

def on_click(event, x, y, p1, p2):
    if event == cv2.EVENT_LBUTTONDOWN:
        show_cell_details(x, y)


images = [f for f in glob.glob("Data/PhC-C2DL-PSC/Sequence_m/*")]

images.sort()

sequence = np.empty(len(images), dtype=list)
cur_image = 0
i = 0

for image_path in images:
    #image_path = "COMP9517 20T2 Group Project Image Sequences/PhC-C2DL-PSC/Sequence 1/t000.tif"

    image = cv2.imread(image_path)

    #processes images to segmented and thresholded cells
    #replace with better segmentation algorithm
    segmented = process_image(image)

    #labels each cell with different values for counting
    flooded = flood(segmented)

    #counts labelled cells, measures bounding boxes and stores in list
    cells = count_cells(flooded)
    sequence[i] = cells

    #cell matching goes here

    i = i + 1
    
cv2.namedWindow('image')
cv2.setMouseCallback('image', on_click)

for i in range(len(images)):
    cur_image = i
    image = cv2.imread(images[i])
    drawn = draw_bounding_box(image, sequence[i])
    print("Number of cells: " + str(len(sequence[i])))
    print("Number of cell divisions: " + str(count_cell_divisions(sequence[i])))

    # plot_two_images(image_path, image, "Original Image", drawn, "Bounding Boxes")
    cv2.imshow("image",drawn)
    cv2.waitKey()
