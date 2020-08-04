import imutils
import cv2
import numpy as np
import argparse

def SegmentNuclei(image, N):
	blur = cv2.GaussianBlur(image,(N,N))

def BackgroundIntensity(image):
	pass

def CellIntensity(image):
	pass

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

	# cv2.imshow("Image", image)
	# print(maxVal)
	# cv2.waitKey(0)
	return image

parser = argparse.ArgumentParser(description='stretching')
parser.add_argument('file', type=str, nargs=1, help='Path to file')

args = parser.parse_args()
filename = args.file[0]

I = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
CS = ContrastStretching(I)

cv2.imshow("CS",CS)
cv2.waitKey()


