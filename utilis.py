import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.future import graph
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage import color
import os
import math
from random import randint

colors = {
    "red"   : (255,0, 0),
    "green" : (0,255,0),
    "blue"  : (0,0,255)
}

def crop(img_mat, black_bg=False):
    img = cv.cvtColor(img_mat, cv.COLOR_BGR2GRAY)

    rows, cols = img.shape

    val = 255

    if black_bg:
        val = 0

    # print(img)

    def left():
        for x in range(cols):
            for y in range(rows):
                if img[y, x] != val:
                    return x

    left = left()
    # cv.circle(img, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv.FILLED)

    def top():
        for y in range(rows):
            for x in range(cols):
                if img[y, x] != val:
                    return y

    top = top()
    # cv.circle(img, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv.FILLED)

    def bottom():
        for y in reversed(range(rows)):
            for x in range(cols):
                if img[y, x] != val:
                    return y

    bottom = bottom()
    # cv.circle(img, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv.FILLED)

    def right():
        for x in reversed(range(cols)):
            for y in range(rows):
                if img[y, x] != val:
                    return x

    right = right()

    # cv.circle(img, (int(left+1), int(top+1)), 8, (0, 255, 255), thickness=-1, lineType=cv.FILLED)

    # cv.imshow("Cropped", img[top:bottom, left:right])
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return img_mat[top:bottom, left:right]

def aggregate(mappings, labels):
    rows, cols = labels.shape

    for x in range(rows):
        for y in range(cols):
            mappings[labels[x, y]] += 1

    # mappings = {k: v for k, v in sorted(mappings.items(), key=lambda item: item[1])}

def set_label(cluster, labels, val):
    rows, cols = labels.shape

    for x in range(rows):
        for y in range(cols):
            if labels[x,y] == cluster:
                labels[x,y] = val

def plot_three_images(figure_title, image1, label1,
                      image2, label2, image3, label3):
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

    plt.tight_layout()
    plt.show()

def plot_two(figure_title, image1, label1, image2, label2):
    # Display the first image
    fig = plt.figure()
    fig.suptitle(figure_title)

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

def plot_four_images(figure_title, image1, label1, image2, label2, image3, label3, image4, label4):
    _, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    ax1, ax2, ax3, ax4 = ax_arr.ravel()

    ax1.imshow(image1)
    ax1.set_title(label1)

    ax2.imshow(image2)
    ax2.set_title(label2)

    ax3.imshow(image3)
    ax3.set_title(label3)

    ax4.imshow(image4)
    ax4.set_title(label4)

    for ax in ax_arr.ravel():
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

def weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])

def pathname(path):
    filename = os.path.basename(path)

    length = len(filename)

    path = path[:-length]

    return path, filename

def sharpen(img):
    _a = 1.25
    _sigma = 1

    I = img

    L = cv.GaussianBlur(I, (5,5), 0)

    H = cv.subtract(I, L) * _a

    O = cv.add(I, H, dtype=8)

    return O

def stretch(image):
    minVal = np.amin(image)
    maxVal = np.amax(image)

    mod = 255.0/(maxVal-minVal)

    table = np.array( [(x-minVal)*mod for x in range(256)] )

    draw = cv.LUT(image, table)

    draw = np.uint8(draw)

    return draw, minVal, maxVal

def show_image(img, title):
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()

def aggregate(labels):
    mappings = dict.fromkeys(np.unique(labels), 0)

    rows, cols = labels.shape

    for x in range(rows):
        for y in range(cols):
            mappings[labels[x, y]] += 1

    # mappings = {k: v for k, v in sorted(mappings.items(), key=lambda item: item[1])}

    return mappings

def displacement(height,width,center1,center2):
    (x1,y1) = center1
    (x2,y2) = center2
    return(math.sqrt((x1-x2)**2 + (y2-y1)**2)/(math.sqrt(height**2+width**2)))

def otsuThreshold(img):
    blur = cv.GaussianBlur(img,(5,5),0)
    _, mask = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)

    return mask

def hMaxima(image):
    intensityArray = FindIntensity(image)
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

def ContrastStretching(image):
    (h, w) = image.shape
    minVal = image.min()
    maxVal = image.max()

    for row in range(0,h):
        for col in range(0,w):
            temp = image[row][col]
            value = (temp-minVal)*(255/(maxVal-minVal))
            image[row][col] = value

    return image

def CheckPixelsEqual(image1,image2):
    (h,w) = image1.shape
    for row in range(0,h):
        for col in range(0,w):
            if(image1[row][col] == image2[row][col]):
                return True
    return False

def hSubtraction(image,h):
    (height,width) = image.shape
    # print("h="+str(h))
    new = image.copy()
    for row in range(0,height):
        for col in range(0,width):
            new[row][col] = image[row][col]-h
            if(new[row][col] < 0):
                new[row][col] = 0
    return ~new

def nFoldDilation(image,h):
    original = image.copy()
    dilated = hSubtraction(image,h)
    structElem = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    # print("entering loop")
    n = 0
    while not(CheckPixelsEqual(original,dilated)):
        dilated = cv.dilate(image,structElem)
        # print("loop"+str(n))
        # cv.imshow("dilate loop"+str(n),dilated)
        n+=1
    return dilated

def checkMatches(matches,matchMatrix):
    Pass = True
    for i in range(len(matches)):
        for j in range(i+1,len(matches)):
            if (matches[i] == matches[j] and matches[i] != -1):
                Pass = False
                # print("match i = "+str(matches[i]))
                # print("match j = "+str(matches[j])) 
                if (matchMatrix[i][int(matches[i])][0] > matchMatrix[j][int(matches[j])][0]):
                    temp = matches[i]
                    if temp+1 == len(matchMatrix[i]):
                        matches[i] = -1
                    else:
                        matches[i] = matchMatrix[i][int(temp+1)][1]
                else:
                    temp = matches[j]
                    if temp+1 == len(matchMatrix[j]):
                        matches[j] = -1
                    else:
                        
                        matches[j] = matchMatrix[j][int(temp+1)][1]

    return matches, Pass

def quicksortMatrix(matchArray):
    if (len(matchArray)<2):
        return matchArray
    low = []
    same = []
    high = []

    pivot = matchArray[randint(0,len(matchArray)-1)][0]

    for matchScore in matchArray:
        if matchScore[0] < pivot:
            low.append(matchScore)
        elif matchScore[0] == pivot:
            same.append(matchScore)
        elif matchScore[0]> pivot:
            high.append(matchScore)

    return quicksortMatrix(low) + same + quicksortMatrix(high)


def printMatchMatrix(matchMatrix, Rows, Cols):
    printMatrix = np.zeros((Rows,Cols))
    for i in range(Rows):
        for j in range(Cols):
            printMatrix[i][j] = matchMatrix[i][j][0]
    for i in range(Rows):
        print(printMatrix[i])