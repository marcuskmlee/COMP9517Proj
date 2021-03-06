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
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage import exposure
import skimage.color
from skimage.io import imsave

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

        self.area = cv.contourArea(self.cnt)

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

    def get_area(self):
        return self.area

    def set_x_velocity(self, new_x_velocity):
        self.x_velocity = new_x_velocity

    def set_y_velocity(self, new_y_velocity):
        self.y_velocity = new_y_velocity

    def set_dividing(self):
        self.dividing = True

    def set_matched(self):
        self.matched = True

class CellManager(object):

    def __init__(self,demo=False):
        self.currImage = 0
        self.sequence = []
        self.demo = demo
        self.blurSize = 5
        self.h = 5
        self.image = []
        self.numCells = 0

    def dataset(self, dataset):
        self.dataset = dataset
        if dataset == "PhC":
            self.h = 16
            self.blurSize = 7
        elif dataset == "Fluo":
            self.h = 10
            self.blurSize = 25
        elif dataset == "DIC":
            self.h = 10
            self.blurSize = 99

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

        temp = cv.bitwise_and(img, mask)

        size = (self.blurSize, self.blurSize)
        temp = cv.GaussianBlur(temp, size, 0)
        # show_image(img, "blurred")

        local_maxima = extrema.local_maxima(temp, connectivity=5)
        label_maxima = label(local_maxima)

        overlay = color.label2rgb(label_maxima, img, alpha=0.7, bg_label=0,
                                bg_color=None, colors=[(1, 0, 0)])

        h_maxima = extrema.h_maxima(temp, self.h)
        label_h_maxima = label(h_maxima)
        overlay_h = color.label2rgb(label_h_maxima, img, alpha=0.7, bg_label=0,
                                    bg_color=None, colors=[(1, 0, 0)])

        # show_image(overlay, "local")
        # if self.demo:
        #     # show_image(overlay, "regional")
        #     cv.imshow("overlay", overlay)
            # imsave("./report/Fluo/hMaxima.png", overlay_h)
            # print(f"hMaxima: {overlay_h.dtype}")

        if self.dataset == "DIC":
            # show_image(overlay, "regional")
            # cv.imwrite("./report/DIC/hmaxima.png", overlay.astype(np.uint8))
            # exit(1)
            return local_maxima.astype(np.uint8)
        
        # print(h_maxima)
        # print(np.amax(h_maxima*255))

        show_image(overlay_h, "")
        imsave(f"./report/{self.dataset}/hMaxima.png", overlay_h)
        # cv.imwrite(f"./report/{self.dataset}/hMaxima.png", h_maxima)

        return h_maxima

    def processImage(self, gray, mask, show=False):
        nuclei = self.hMaxima(gray, mask)
        mask = clear_border(mask)
        # mask = clear_border(mask)
        if self.demo:
            # show_image(mask, "Remove contours on edge")
            cv.imwrite(f"./report/{self.dataset}/remove-edges.png", mask)

        # nuclei = self.hMaxima(gray, mask)
        self.image = gray

        # if True:
        #     plot_two("Original vs nuclei", gray, "Original", nuclei, "Nuclei segmented")


        #counts labelled cells, measures bounding boxes and stores in list

        pred_count, sequence = self.count_cells(mask, nuclei)

        self.sequence.append(sequence)
        self.matchCells(gray)
        color = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        drawn = self.draw_bounding_box(color)

        if True:
            cv.imwrite(f"./report/{self.dataset}/bounding_boxes.png", drawn)
            self.show(drawn)

        print(f"Processed image: {self.currImage} with {pred_count} cells")
        self.currImage = self.currImage + 1

        return pred_count

    def show(self, drawn):
        cv.imshow("Bounding Box", drawn)
        cv.waitKey(0)

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
            if (self.in_image(i, cell_id)):
                total = total + distance.euclidean(self.get_cell(i, cell_id).get_centre(), self.get_cell(i + 1, cell_id).get_centre())
        return total

    def calculate_net_distance(self, cell_id):
        for i in range(self.currImage):
            if self.in_image(i, cell_id):
                cell1 = self.get_cell(i, cell_id)
                cell2 = self.get_cell(self.currImage, cell_id)
                return distance.euclidean(cell1.get_centre(), cell2.get_centre())
        return 0

    def draw_cell_track(self, image, cell):
        colour = (0, 255, 0)
        thickness = 1
        for i in range(self.currImage):
            if (self.in_image(i, cell.get_id())):
                prev_cell = self.get_cell(i, cell.get_id())
                next_cell = self.get_cell(i+1, cell.get_id())
                if (next_cell != None and prev_cell != None):
                    image = cv.line(image, prev_cell.get_centre(), next_cell.get_centre(), colour, thickness)
        return image

    def draw_tracks(self, image):
        for cell in self.sequence[self.currImage]:
            image = self.draw_cell_track(image, cell)
        return image

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

    def add_cell(self, cnt):
        self.numCells += 1
        return Cell(self.numCells, cnt)

    def count_cells(self, mask, nuclei):
        _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        sequence = []

        if self.dataset == "DIC":
            _, cnts, _ = cv.findContours(nuclei, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            for i, c in enumerate(cnts):
                rect = cv.boundingRect(c)
                x, y, w, h = rect

                cell = self.add_cell(c)
                sequence.append(cell)

            return len(cnts), sequence

        for i, c in enumerate(contours):
            rect = cv.boundingRect(c)
            x, y, w, h = rect

            # print(rect)

            # drawn = self.image.copy()
            # colour = (255, 0, 0)
            # cv.rectangle(drawn, (int(x), int(y)), (int(w+x), int(y+h)), colour, 1)
            # plot_two("zoom", drawn, "contour", drawn[y:y+h, x:x+w], "zoom")

            if self.count_peaks(nuclei[y:y+h, x:x+w]) > 1:
                show_image(self.image[y:y+h, x:x+w], "Cluster")

                drawn = self.image.copy()
                colour = (255, 0, 0)
                cv.rectangle(drawn, (int(x), int(y)), (int(w+x), int(y+h)), colour, 1)
                # show_image(drawn, "cluster")

                rect = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                poly = np.array([rect], dtype=np.int32)
                clusterMask = np.zeros(mask.shape, dtype=np.uint8)
                cv.fillPoly(clusterMask, poly, 255)
                # plot_two("Drawn", self.image, "Image", clusterMask, "Cluster")
                drawn = cv.bitwise_and(nuclei, clusterMask)
                # show_image(drawn, "drawn")
                # print(f"drawn: {drawn.dtype}")
                # cv.imwrite(f"./report/Fluo/cluster{self.numCells}.png", drawn)

                _, nContours, _ = cv.findContours(drawn, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                for cnt in nContours:
                    cell = self.add_cell(cnt)
                    sequence.append(cell)

                # expanded = expand_labels(nuclei, distance=self.average_radius())
                # plot_two("Drawn", nuclei, "Image", expanded, "Cluster")
            else:
                cell = self.add_cell(c)
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
            cv.putText(drawn, f"ID: {cell.get_id()}", cell.get_centre(), cv.FONT_HERSHEY_DUPLEX, 0.5, colour, 1, cv.LINE_AA)

        drawn = self.draw_tracks(drawn)

        return drawn

    def matchCells(self,image):
        if self.currImage == 0:
            return

        h,w = image.shape
        prevCells = self.sequence[self.currImage-1]
        currCells = self.sequence[self.currImage]
        numPrev = len(prevCells)
        numCurr = len(currCells)
        matchingMatrix = np.full((numCurr,numPrev,2),100, dtype=float)
        # minMatch = np.full((numCurr,2),100)
        for i in range(numCurr):
            for j in range(numPrev):
                displace = displacement(h,w,currCells[i].centre, prevCells[j].centre)
                # print(100*displace)
                # diffArea = abs(currCells[i].area - prevCells[j].area)
                matchingMatrix[i][j][0] = displace
                # print(displace)
                # print("matchingmatrix["+str(i)+"]["+str(j)+"]="+str(matchingMatrix[i][j][0]))
                matchingMatrix[i][j][1] = j


        # print("Matching Matrix:")
        # print(matchingMatrix)
        # printMatchMatrix(matchingMatrix, numCurr, numPrev)

        sortedMatrix = np.zeros((numCurr,numPrev,2))

        # print("Matching Matrix:")
        # printMatchMatrix(matchingMatrix, numCurr, numPrev)

        sortedMatrix = np.zeros((numCurr,numPrev,2))

        for i in range(numCurr):
            sortedMatrix[i] = quicksortMatrix(matchingMatrix[i])

        # print("Original")
        # print(matchingMatrix)
        # print("Sorted")
        # print(sortedMatrix)

        matches = np.zeros(numCurr)
        # print(sortedMatrix[0][0])
        # print(len(sortedMatrix[0][0]))
        for i in range(numCurr):
            matches[i] = sortedMatrix[i][0][1]

        # print("matches:")
        # print(matches)
        matches, success = checkMatches(matches, sortedMatrix)
        while not success:
            matches, success = checkMatches(matches, sortedMatrix)

        for i in range(numCurr):
            prev = self.sequence[self.currImage-1][int(matches[i])]
            cell = self.sequence[self.currImage][i]

            if matchingMatrix[i][int(matches[i])][0] < 0.2:
                # print("rejected")
                matches[i] = -1

            if (matches[i] != -1):
                cell = self.sequence[self.currImage][i]
                cell.set_id(self.sequence[self.currImage-1][int(matches[i])].get_id())
                if prev.get_area() * 1.3 >= cell.get_area():
                    cell.set_dividing()
                cell.set_id(prev.get_id())
