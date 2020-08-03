#Group A COMP9517 Project

import imutils
import cv2
import numpy as np

#QUESTION 1

#Step 1:
#Read images from a given folder

#Step 2:
#Read first image from a given folder

#Step 3:
#Initialise some form of tracking classifier

#Step 4:
#Apply the Kalman filter to the loaded image

#Step 5:
#Apply/draw the boundary boxes


#Step 6:
#Track the motion of the particles

#Step 7:
#Draw the trajectory from the tracked motion

#Step 8:
#Count the number of cells (number of boundary boxes)

#Step 9:
#Output this number

#Step 10:
#DONE

#QUESTION 2
def maxNeighbourhood(N,image,x,y):
	# A function that creates a neighbourhood/box of size NxN and returns the maximum pixel value in that neighbourhood
	(h,w) = image.shape
	top = x-N/2
	if top < 0:
		top = 0
	bottom = x+N/2
	if bottom > h:
		bottom = h
	left = y-N/2
	if left < 0:
		left = 0
	right = y+N/2
	if right > w:
		right = w
	neighbourhood = image[int(top):int(bottom), int(left):int(right)]
	# cv2.imshow("neigh", neighbourhood)
	# cv2.waitKey(0)
	return neighbourhood.max()

def minNeighbourhood(N,image,x,y):
	#A function that creates a neighbourhood/box of size NxN and returns the min pixel value
	(h,w) = image.shape
	top = x-N/2
	if top < 0:
		top = 0
	bottom = x+N/2
	if bottom > h:
		bottom = h
	left = y-N/2
	if left < 0:
		left = 0
	right = y+N/2
	if right > w:
		right = w
	neighbourhood = image[int(top):int(bottom), int(left):int(right)]
	# cv2.imshow("neigh", neighbourhood)
	# cv2.waitKey(0)
	return neighbourhood.min()

def ContrastStretch(image, N):
	(h, w) = image.shape
	A = image.copy()
	B = image.copy()
	for x in range(0,h):
		for y in range(0,w):
			A[x][y] = maxNeighbourhood(N,I,x,y)

	for x in range(0,h):
		for y in range(0,w):
			B[x][y] = minNeighbourhood(N,A,x,y)

	#Invert the colours so that the image comes out with a white background similar to the original
	O = ~(cv2.subtract(B,I))
	return O

def DetectCells(image):
	(h,w,d) = image.shape
	# kernel = np.ones((5,5),np.float32)/25
	# dst = cv2.bilateralFilter(image,-1,15,10)
	
	# return dst
	

I = cv2.imread("COMP9517 20T2 Group Project Image Sequences/DIC-C2DH-HeLa/Sequence 1/t000.tif", cv2.IMREAD_GRAYSCALE)	
CS = ContrastStretch(I,45)
# kernel = np.ones((40,40), np.uint(8))
# TH = cv2.morphologyEx(I, cv2.MORPH_TOPHAT, kernel)
# cv2.imshow("I",I)
# cv2.imshow("CS", CS)
# cv2.imshow("colorCS", cv2.cvtColor(CS, cv2.COLOR_GRAY2BGR))
# cv2.waitKey()
Cells = DetectCells(cv2.cvtColor(CS, cv2.COLOR_GRAY2BGR))
# cv2.imshow("Cells", CS)
# cv2.waitKey()





