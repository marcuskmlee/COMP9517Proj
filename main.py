# import the necessary packages
import cv2
import numpy as np
import math 
import glob 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.spatial import distance

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

from cell import Cell, CellManager
import argparse

parser = argparse.ArgumentParser(description='Comp9517 Project')
parser.add_argument('--DIC', nargs='?', default=False, const=True,
    help='Use DIC-C2DH-HeLa dataset')
parser.add_argument('--Flou', nargs='?', default=False, const=True,
    help='Use Flou-N2DL-HeLa dataset')
parser.add_argument('--PhC', nargs='?', default=False, const=True, 
    help='Use PhC-C2DL-HeLa dataset')

args = parser.parse_args()

manager = CellManager()

def run_PhC():
    sequences = ["Sequence_2", "Sequence_1", "Sequence_3", "Sequence_4"]
    for folder in sequences:    
        images = [f for f in glob.glob(f"./Data/{datasets[2]}/{folder}/*")]
        images.sort()

        for image_path in images:
            #image_path = "COMP9517 20T2 Group Project Image Sequences/PhC-C2DL-PSC/Sequence 1/t000.tif"

            image = cv2.imread(image_path)

            #processes images to segmented and thresholded cells
            #replace with better segmentation algorithm
            mask = PhC.preprocess_image(image)
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

            #counts labelled cells, measures bounding boxes and stores in list
            pred_count = manager.count_cells(mask)

            i = i + 1

datasets = ["DIC-C2DH-HeLa", "Flou-N2DL-HeLa", "PhC-C2DL-PSC"]
cv2.namedWindow('image')
cv2.setMouseCallback('image', on_click)

sequence = np.empty(len(images), dtype=list)
cur_image = 0
i = 0

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