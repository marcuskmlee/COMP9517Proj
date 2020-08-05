# import the necessary packages
import cv2
import numpy as np
import math 
import glob 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift

from skimage.segmentation import flood_fill

from utilis import *
from PIL import Image
import math

import PhC
import Flou

from cell import Cell
import argparse

parser = argparse.ArgumentParser(description='Comp9517 Project')
parser.add_argument('--DIC', nargs='?', default=False, const=True,
    help='Use DIC-C2DH-HeLa dataset')
parser.add_argument('--Flou', nargs='?', default=False, const=True,
    help='Use Flou-N2DL-HeLa dataset')
parser.add_argument('--PhC', nargs='?', default=False, const=True, 
    help='Use PhC-C2DL-HeLa dataset')

args = parser.parse_args()

def add_cell(_id, x, y, cnt):
    cells.append(Cell(_id, x, y, cnt))

def count_cells(mask):
    _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    for i, c in enumerate(contours):
        add_cell(i, c)

    colour = (0, 255, 0)
    
    return len(contours)

def draw_bounding_box(image, cells):
    drawn = image.copy()

    for cell in cells:
        colour = (0, 255, 0)
        if cell.dividing:
            colour = (0, 0, 255)

        cv2.rectangle(drawn, (cell.get_x_min(), cell.get_y_min()), (cell.get_x_max(), cell.get_y_max()), colour, 1)
        cv.circle(drawing, center, 1, color, 2)
    
    return drawn

def run_PhC():
    sequences = ["Sequence_2", "Sequence_1", "Sequence_3", "Sequence_4"]
    for folder in sequences:    
        images = [f for f in glob.glob(f"./Data/{datasets[2]}/{folder}/*")]
        images.sort()
        cells = []

        sequence = np.empty(len(images), dtype=list)
        i = 0

        for image_path in images:
            #image_path = "COMP9517 20T2 Group Project Image Sequences/PhC-C2DL-PSC/Sequence 1/t000.tif"

            image = cv2.imread(image_path)

            #processes images to segmented and thresholded cells
            #replace with better segmentation algorithm
            mask = PhC.preprocess_image(image)
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

            show_image(mask, "Mask")

            #counts labelled cells, measures bounding boxes and stores in list
            pred_count = count_cells(mask, image_path)

datasets = ["DIC-C2DH-HeLa", "Flou-N2DL-HeLa", "PhC-C2DL-PSC"]

if args.DIC:
    run_DIC()
elif args.Flou:
    run_Flou()
elif args.PhC:
    run_PhC()
else:
    run_DIC()
    run_Flou()
    run_PhC()