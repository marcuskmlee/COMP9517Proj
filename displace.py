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

def displacement(height, width, center1, center2):
	(x1,y1) = center1
	(x2,y2) = center2
	return(math.sqrt((x2-x1)**2 + (y2-y1)**2)/(math.sqrt(height**2+width**2)))

