import imutils
import cv2
import numpy as np 
import argparse

def maxNeighbourhood(N,image,x,y):
	# A function that creates a neighbourhood/box of size NxN and returns the maximum pixel value in that neighbourhood
	(h,w,d) = image.shape
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
	(h,w,d) = image.shape
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

#Input from user to determine the image, neighbourhood size, and method
parser = argparse.ArgumentParser(description='stretching')
parser.add_argument('file', type=str, nargs=1, help='Path to file')
parser.add_argument('N', type=int, nargs=1, help='Value of N')
parser.add_argument('M', type=int, nargs=1, help='Value of M')

args = parser.parse_args()
filename = args.file[0]
N = args.N[0]
M = args.M[0]

I = cv2.imread(filename)

# print("Enter value of N:")
# N = int(input())
# print("Enter value of M:")
# M = int(input())

#Reading the images
(h, w, d) = I.shape
A = I.copy()
B = I.copy()

if M == 0:
	#If M == 0 then we perform max-filtering, then min-filtering
	for x in range(0,h):
		for y in range(0,w):
			for z in range(0,d):
				A[x][y][z] = maxNeighbourhood(N,I,x,y)

	for x in range(0,h):
		for y in range(0,w):
			for z in range(0,d):
				B[x][y][z] = minNeighbourhood(N,A,x,y)

	#Invert the colours so that the image comes out with a white background similar to the original
	O = ~(cv2.subtract(B,I))

elif M == 1:
	#If m == 1 then we perform min-filtering, then max-filtering
	for x in range(0,h):
		for y in range(0,w):
			for z in range(0,d):
				A[x][y][z] = minNeighbourhood(N,I,x,y)

	for x in range(0,h):
		for y in range(0,w):
			for z in range(0,d):
				B[x][y][z] = maxNeighbourhood(N,A,x,y)

	O = (cv2.subtract(I,B))

#The following section is for writing/showing the original image, the output image, and the images created in the method


cv2.imshow("I",I)
# cv2.imshow("A",A)
# cv2.imshow("B",B)
cv2.imshow("O",O)
# cv2.imwrite("B"+":N="+str(N)+img,B)
# cv2.imwrite("O"+":N="+str(N)+img,O)
cv2.waitKey(0)
	