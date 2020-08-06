import cv2 as cv
import numpy as np
import math

from scipy.spatial import distance

from skimage.measure import label
from skimage import data
from skimage import color
from skimage.morphology import extrema
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage import exposure

from utilis import *

from segment import *
class Cell(object):
    def __init__(self, i, cnt):
        self.id = i
        self.cnt = cnt
        self.rect = cv.boundingRect(cnt)
        self.centre, self.radius = cv.minEnclosingCircle(cnt)

        self.x_velocity = 0
        self.y_velocity = 0

        self.dividing = False
        self.matched = False
        self.inFrame = True

        self.area = cv.contourArea(self.contours)

    def __str__(self):
        return "Cell id: " + str(self.id) + " x range: " + str(self.x) + "-" + str(self.w) + " y range: " + str(self.y) + "-" + str(self.h)

    def contains(self, px, py):
        x, y, w, h = self.rect
        if x < px < x+w:
            if y < py < y+h:
                return True
        return False

    def is_dividing(self):
        return self.dividing

    def get_matched(self):
        return self.matched

    def get_centre(self):
        x, y = self.centre
        return (int(x), int(y))

    def get_radius(self):
        return self.radius

    def get_id(self):
        return self.id

    def get_rect(self):
        return self.rect

    def get_x_velocity(self):
        return self.x_velocity

    def get_y_velocity(self):
        return self.y_velocity

    def set_id(self, new_id):
        self.id = new_id

    def set_x_velocity(self, new_x_velocity):
        self.x_velocity = new_x_velocity

    def set_y_velocity(self, new_y_velocity):
        self.y_velocity = new_y_velocity

    def set_dividing(self):
        self.dividing = True

    def set_matched(self):
        self.matched = True

class CellManager(object):

    def __init__(self, dataset, demo=False):
        self.cells = []
        self.currImage = 0
        self.sequence = []
        self.demo = demo
        self.blurSize = 0
        self.h = 5
        self.image = []

        if dataset == "PhC":
            self.h = 16
            self.blurSize = 7
        elif dataset == "Fluo":
            self.h = 10
            self.blurSize = 19

    def count_cell_divisions(self, cells):
        count = 0
        for cell in cells:
            if (cell.is_dividing()):
                count = count + 1
        return count

    def hMaxima(self, img, mask):
        # maximaKernel = np.ones((5,5),np.uint8)
        # maximaKernel[2,2] = 0

        # gray = cv.GaussianBlur(gray, (5,5), 0)
        # gray = cv.bitwise_and(gray, mask)
        # if self.demo:
        #     show_image(gray, "Remove background")

        # maxima = cv.dilate(gray, maximaKernel, iterations = 5)
        # maxima = cv.compare(gray, maxima, cv.CMP_GE)
        # if self.demo:
        #     plot_two("Find Maxima", gray, "Original", maxima, "Background subtracted")

        # minima = cv.erode(gray, maximaKernel, iterations = 1)
        # minima = cv.compare(gray, minima, cv.CMP_GT)
        # maxima = cv.bitwise_and(maxima, minima)
        # maxima = cv.GaussianBlur(maxima, (5,5), 0)

        size = (self.blurSize, self.blurSize)
        img = cv.GaussianBlur(img, size, 0)
        img = cv.bitwise_and(img, mask)

        local_maxima = extrema.local_maxima(img, connectivity=5)
        label_maxima = label(local_maxima)
        overlay = color.label2rgb(label_maxima, img, alpha=0.7, bg_label=0,
                                bg_color=None, colors=[(1, 0, 0)])

        h_maxima = extrema.h_maxima(img, self.h)
        label_h_maxima = label(h_maxima)
        overlay_h = color.label2rgb(label_h_maxima, img, alpha=0.7, bg_label=0,
                                    bg_color=None, colors=[(1, 0, 0)])

        # show_image(overlay, "local")
        if self.demo:
            show_image(overlay_h, "regional")

        return h_maxima

    def processImage(self, gray, mask, show=False):
        nuclei = self.hMaxima(gray, mask)
        self.image = gray

        mask = clear_border(mask)

        if self.demo:
            plot_two("Original vs nuclei", gray, "Original", nuclei, "Nuclei segmented")

        #counts labelled cells, measures bounding boxes and stores in list
        pred_count, sequence = self.count_cells(mask, nuclei)

        self.sequence.append(sequence)
        self.matchCells(img)
        drawn = self.draw_bounding_box(img)

        if show:
            self.show(drawn)

        print(f"Processed image: {self.currImage}")
        self.currImage = self.currImage + 1

    def show(self, drawn):
        cv.imshow("Bounding Box", drawn)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def in_image(self, image_no, cell_id):
        if (image_no < 0 or image_no >= len(self.sequence)):
            return False
        for cell in self.sequence[image_no]:
            if (cell.get_id() == cell_id):
                return True
        return False

    def get_cell(self, image_no, cell_id):
        for cell in self.sequence[image_no]:
            if (cell.get_id() == cell_id):
                return cell
        return None

    def calculate_speed(self, cell_id):
        if self.in_image(self.currImage - 1, cell_id):
            return distance.euclidean(self.get_cell(self.currImage-1, cell_id).get_centre(), self.get_cell(self.currImage, cell_id).get_centre())
        return 0

    def calculate_total_distance(self, cell_id):
        total = 0
        for i in range(self.currImage - 1):
            if (in_image(i, cell_id)):
                total = total + distance.euclidean(self.get_cell(i, cell_id).get_centre(), self.get_cell(i + 1, cell_id).get_centre())
        return total

    def calculate_net_distance(self, cell_id):
        for i in range(self.currImage):
            if self.in_image(i, cell_id):
                cell1 = self.get_cell(i, cell_id)
                cell2 = self.get_cell(self.currImage, cell_id)
                return distance.euclidean(cell1.get_centre(), cell2.get_centre())
        return 0

    def show_cell_details(self, x, y):
        for cell in self.sequence[self.currImage]:
            cell_id = cell.get_id()
            if (cell.contains(x, y) and self.in_image(self.currImage, cell_id)):
                speed = self.calculate_speed(cell_id)
                print("Speed: " + str(speed))
                total_distance = self.calculate_total_distance(cell_id)
                print("Total Distance: " + str(total_distance))
                net_distance = self.calculate_net_distance(cell_id)
                print("Net Distance: " + str(net_distance))
                confinement = 0
                if net_distance != 0:
                    confinement = (total_distance / net_distance)
                print("Confinement Ratio: " + str(confinement))

    def add_cell(self, _id, cnt):
        self.cells.append(Cell(_id, cnt))
        return Cell(_id, cnt)

    def count_cells(self, mask, nuclei):
        _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        sequence = []
        for i, c in enumerate(contours):
            rect = cv.boundingRect(c)
            x, y, w, h = rect

            # drawn = self.image.copy()
            # colour = (255, 0, 0)
            # cv.rectangle(drawn, (int(x), int(y)), (int(w+x), int(y+h)), colour, 1)
            # plot_two("zoom", drawn, "contour", drawn[y:y+h, x:x+w], "zoom")

            if self.count_peaks(nuclei[y:y+h, x:x+w]) > 1:
                expanded = expand_labels(nuclei, distance=self.average_radius())

                poly = np.array(rect, dtype=np.int32)
                img = np.zeros((800, 800), np.int8)
                cv2.fillPoly(img, poly, 255)
                imshow(img, cmap="gray")

                drawn = self.image.copy()
                colour = (255, 0, 0)
                cv.rectangle(drawn, (int(x), int(y)), (int(w+x), int(y+h)), colour, 1)
                show_image(drawn, "cluster")

            cell = self.add_cell(i, c)
            sequence.append(cell)

        colour = (0, 255, 0)

        return len(contours), sequence

    def count_peaks(self, mask):
        _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        return len(contours)

    def average_radius(self, i):
        i = self.currImage
        avg = 0
        for cell in self.sequence[i]:
            avg += cell.get_radius()

        return int(avg/len(sequence[i]))

    def draw_bounding_box(self,image):
        drawn = image.copy()

        for cell in self.sequence[self.currImage]:
            colour = (0, 255, 0)
            if cell.is_dividing():
                colour = (0, 0, 255)

            x, y, w, h = cell.get_rect()
            cv.rectangle(drawn, (int(x), int(y)), (int(w+x), int(y+h)), colour, 1)
            cv.circle(drawn, cell.get_centre(), 1, colour, 2)

        return drawn

    def matchCells(self,image):
        (h,w,d) = image.shape
        prevCells = self.sequence[self.currImage-1]
        currCells = self.sequence[self.currImage]
        numPrev = len(prevCells)
        numCurr = len(currCells)
        matchingMatrix = np.full((numCurr,numPrev,2),100)
        # minMatch = np.full((numCurr,2),100)
        for i in range(numCurr):
            for j in range(numPrev):
                displace = displacement(h,w,currCells[i].centre, prevCells[j].centre)
                diffArea = abs(currCells[i].area - prevCells[j].area)
                matchingMatrix[i][j][0] = displace+diffArea
                matchingMatrix[i][j][1] = j
                # if (displacement+diffArea < minMatch[i][0]):
                #     minMatch[i][0] = displacement+diffArea
                #     minmatch[i][1] = j
        # print("Matching Matrix Pre-Sort")
        # print(matchingMatrix)
        # matchingMatrix.sort(axis = 1)
        # print ((matchingMatrix[0]))
        # print("Matching Matrix Post-Sort")
        # print(matchingMatrix)
        # for i in range(numCurr):
        #     for j in range(numPrev):
        #         if (onlyMatch(matchingMatrix, matchingMatrix[i][j][1])):
        #             sequence[i].set_id(sequence[])
        sortedMatrix = np.zeros((numCurr,numPrev,2))
        for i in range(numCurr):
            sortedMatrix[i] = quicksortMatrix(matchingMatrix[i])

        print("Original")
        print(matchingMatrix)
        print("Sorted")
        print(sortedMatrix)

        matches = np.zeros(numCurr)
        for i in range(numCurr):
            matches[i] = sortedMatrix[i][0][1]

        print("matches:")
        print(matches)
        matches, success = checkMatches(matches, sortedMatrix)
        while not success:
            matches, success = checkMatches(matches, sortedMatrix)

        for i in range(numCurr):
            if (matches[i] != -1):
                self.sequence[self.currImage][i].set_id(self.sequence[self.currImage][int(matches[i])].get_id())
