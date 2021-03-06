import imutils
import cv2 as cv
import numpy as np
import argparse
from utilis import plot_two

from skimage.measure import label
from skimage import data
from skimage import color
from skimage.morphology import extrema
from skimage import exposure

import matplotlib.pyplot as plt

import PhC

from utilis import show_image

def SegmentNuclei(image, N):
	blur = cv.GaussianBlur(image,(N,N))

def ContrastStretching(image):
	(h, w) = image.shape
	minVal = image.min()
	maxVal = image.max()

	# print ("width={}, height={}, depth={}".format(h,w,d))

	for row in range(0,h):
		for col in range(0,w):
			temp = image[row][col]
			value = (temp-minVal)*(255/(maxVal-minVal))
			image[row][col] = value

	# cv.imshow("Image", image)
	# print(maxVal)
	# cv.waitKey(0)
	return image

def hMaxima(image, intensityArray):
	nBg = 0
	sumBg = 0
	nCells = 0
	sumCells = 0
	for i in range(0,256):
		if(intensityArray[i] < 1000):
			nBg += intensityArray[i]
			sumBg += intensityArray[i]*i 
		else:
			nCells += intensityArray[i]
			sumCells += intensityArray[i]*i

	Beta = 0.6
	AveBg = sumBg/nBg
	AveCells = sumCells/nCells
	h = (Beta/2)*((AveCells-AveBg)**2)/(AveCells+AveBg)
	return h

def FindIntensity(image):
	intensity = np.zeros(256, dtype=int)
	(h, w) = image.shape
	for row in range(0,h):
		for col in range(0,w):
			intensity[image[row][col]] += 1

	return intensity

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
	# cv.imshow("neigh", neighbourhood)
	# cv.waitKey(0)
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
	# cv.imshow("neigh", neighbourhood)
	# cv.waitKey(0)
	return neighbourhood.min()

def MinMaxFilter(image, N, M):
	(h,w) = image.shape
	A = image.copy()
	B = image.copy()

	if M == 0:
	#If M == 0 then we perform max-filtering, then min-filtering
		for x in range(0,h):
			for y in range(0,w):
				A[x][y] = maxNeighbourhood(N,image,x,y)

		for x in range(0,h):
			for y in range(0,w):
				B[x][y] = minNeighbourhood(N,A,x,y)

		#Invert the colours so that the image comes out with a white background similar to the original
		O = ~(cv.subtract(B,I))

	elif M == 1:
		#If m == 1 then we perform min-filtering, then max-filtering
		for x in range(0,h):
			for y in range(0,w):
				A[x][y] = minNeighbourhood(N,image,x,y)

		for x in range(0,h):
			for y in range(0,w):
				B[x][y] = maxNeighbourhood(N,A,x,y)

		O = (cv.subtract(I,B))

	return O


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='stretching')
	parser.add_argument('file', type=str, nargs=1, help='Path to file')

	args = parser.parse_args()
	filename = args.file[0]

	img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
	mask = PhC.preprocess(img)

	img = cv.GaussianBlur(img, (5,5), 0)
	img = cv.bitwise_and(img, mask)

	local_maxima = extrema.local_maxima(img)
	label_maxima = label(local_maxima)
	overlay = color.label2rgb(label_maxima, img, alpha=0.7, bg_label=0,
							bg_color=None, colors=[(1, 0, 0)])

	h = 16
	h_maxima = extrema.h_maxima(img, h)
	label_h_maxima = label(h_maxima)
	overlay_h = color.label2rgb(label_h_maxima, img, alpha=0.7, bg_label=0,
								bg_color=None, colors=[(1, 0, 0)])

	show_image(overlay, "local")
	show_image(overlay_h, "regional")

	# fig, ax = plt.subplots(1, 3, figsize=(15, 5))

	# ax[0].imshow(img, cmap='gray')
	# ax[0].set_title('Original image')
	# ax[0].axis('off')

	# ax[1].imshow(overlay)
	# ax[1].set_title('Local Maxima')
	# ax[1].axis('off')

	# ax[2].imshow(overlay_h)
	# ax[2].set_title('h maxima for h = %.2f' % h)
	# ax[2].axis('off')
	# plt.show()

	# CS = ContrastStretching(I)
	# minmax = MinMaxFilter(CS, 20, 1)
	# intensityArray = FindIntensity(minmax)
	# # print(intensity)
	# h = hMaxima(minmax, intensityArray)

	# print(h)


	# plot_two("H Maxima", I, "Original", minmax, "MinMaxed")