# import the necessary packages
import cv2 as cv
import numpy as np
import math 
import glob 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.spatial import distance

from utilis import *
from PIL import Image
import math

import PhC
import Fluo

from cell import Cell, CellManager
import argparse

parser = argparse.ArgumentParser(description='Comp9517 Project')
parser.add_argument('--DIC', nargs='?', default=False, const=True,
    help='Use DIC-C2DH-HeLa dataset')
parser.add_argument('--Fluo', nargs='?', default=False, const=True,
    help='Use Fluo-N2DL-HeLa dataset')
parser.add_argument('--PhC', nargs='?', default=False, const=True, 
    help='Use PhC-C2DL-HeLa dataset')
parser.add_argument('--demo', nargs='?', default=False, const=True, 
    help='Use PhC-C2DL-HeLa dataset')

args = parser.parse_args()

def run_PhC():
    sequences = ["Sequence_1", "Sequence_2", "Sequence_3", "Sequence_4"]
    manager = CellManager(demo=args.demo, dataset="PhC")
    for folder in sequences:    
        images = [f for f in glob.glob(f"./Data/{datasets[2]}/{folder}/*")]
        images.sort()

        for image_path in images:
            #image_path = "COMP9517 20T2 Group Project Image Sequences/PhC-C2DL-PSC/Sequence 1/t000.tif"
            img = cv.imread(image_path, cv.COLOR_BGR2GRAY)

            #processes images to segmented and thresholded cells
            #replace with better segmentation algorithm
            mask = PhC.preprocess(img)
            if args.demo:
                plot_two("Mask", img, "Original", mask, "Mask")

            show = images.index(image_path) == len(images)-1
            manager.processImage(img, mask, show=show)

def run_DIC():
    sequences = ["Sequence_1", "Sequence_2", "Sequence_3", "Sequence_4"]
    masks = ["Sequence_1_Preds", "Sequence_2_Preds", "Sequence_3_Preds", "Sequence_4_Preds"]
    dataset = zip(sequences, masks)
    manager = CellManager(demo=args.demo, dataset="DIC")
    for image_folder, mask_folder in dataset:    
        images = [f for f in glob.glob(f"./Data/{datasets[0]}/{image_folder}/*")]
        masks = [f for f in glob.glob(f"./Data/{datasets[0]}/{mask_folder}/*")]
        images.sort()
        masks.sort()

        data = zip(images, masks)
        for image_path, mask_path in data:
            img = cv.imread(image_path, cv.COLOR_BGR2GRAY)
            mask = cv.imread(mask_path, cv.COLOR_BGR2GRAY)

            kernel = np.ones((10,10),np.uint8)
            closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

            show = images.index(image_path) == len(images)-1
            manager.processImage(img, mask, show=show)

def run_Fluo():
    sequences = ["Sequence_1", "Sequence_2", "Sequence_3", "Sequence_4"]
    manager = CellManager(demo=args.demo, dataset="Fluo")
    for folder in sequences:
        images = [f for f in glob.glob(f"./Data/{datasets[1]}/{folder}/*")]
        images.sort()

        for image_path in images:
            img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

            if args.demo:
                show_image(img, "Original")

            mask, img = Fluo.preprocess_image(img)

            if args.demo:
                plot_two("Mask", img, "Original", mask, "Mask")

            show = images.index(image_path) == len(images)-1
            manager.processImage(img, mask, show=show)

def on_click(event, x, y, p1, p2):
    if event == cv.EVENT_LBUTTONDOWN:
        manager.show_cell_details(x, y)

datasets = ["DIC-C2DH-HeLa", "Fluo-N2DL-HeLa", "PhC-C2DL-PSC"]
cv.namedWindow('Bounding Box')
cv.setMouseCallback('Bounding Box', on_click)

cur_image = 0
i = 0

if args.DIC:
    run_DIC()
elif args.Fluo:
    run_Fluo()
elif args.PhC:
    run_PhC()
else:
    run_DIC()
    run_Fluo()
    run_PhC()