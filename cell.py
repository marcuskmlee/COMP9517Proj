import cv2 as cv
import numpy as np

from scipy.spatial import distance

from utilis import *

import PhC
import Flou
class Cell(object):
    def __init__(self, i, cnt):
        self.id = i
        self.cnt = cnt
        rect = cv.boundingRect(cnt)
        self.center, _ = cv.minEnclosingCircle(cnt)

        self.x = int(rect[0])
        self.y = int(rect[1])
        self.w = int(x + rect[2])
        self.h = int(h + rect[3])

        self.x_velocity = 0
        self.y_velocity = 0

        self.dividing = False
        self.matched = False
        self.inFrame = True

    def __str__(self):
        return "Cell id: " + str(self.id) + " x range: " + str(self.x) + "-" + str(self.y) + " y range: " + str(self.w) + "-" + str(self.h)

    def update_bound(self, x, y):
        if (x < self.x):
            self.x = x
        if (x > self.y):
            self.y = x
        if (y < self.w):
            self.w = y
        if (y > self.h):
            self.h = y

    def contains(self, x, y):
        if (x < self.y and x > self.x):
            if (y < self.h and y > self.w):
                return True
        return False

    def get_dividing(self):
        return self.dividing

    def get_matched(self):
        return self.matched
    
    def get_centre(self):
        return (int((self.x + self.y)/2),int((self.w + self.h)/2)) 

    def get_id(self):
        return self.id

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_w(self):
        return self.w

    def get_h(self):
        return self.h

    def get_x_velocity(self):
        return self.x_velocity

    def get_y_velocity(self):
        return self.y_velocity
    
    def set_id(self, new_id):
        self.id = new_id

    def set_x(self, new_x):
        self.x = new_x

    def set_y(self, new_x):
        self.y = new_x

    def set_w(self, new_y):
        self.w = new_y

    def set_h(self, new_y):
        self.h = new_y

    def set_x_velocity(self, new_x_velocity):
        self.x_velocity = new_x_velocity

    def set_y_velocity(self, new_y_velocity):
        self.y_velocity = new_y_velocity

    def set_dividing(self):
        self.dividing = True

    def set_matched(self):
        self.matched = True

class CellManager(object):

    def __init__(self):
        self.cells = []
        self.currImage = 0
        self.sequence = []

    def count_cell_divisions(self, cells):
        count = 0
        for cell in cells:
            if (cell.get_dividing()):
                count = count + 1
        return count

    def processImage(self, filename):
        img = cv.imread(filename, cv.COLOR_BGR2GRAY)

        #processes images to segmented and thresholded cells
        #replace with better segmentation algorithm
        mask, img = PhC.preprocess(img)

        h = hMaxima(img)
        hmax = nFoldDilation(img, h)

        show_image(hmax, "Dilated")
        return

        #counts labelled cells, measures bounding boxes and stores in list
        pred_count, sequence = manager.count_cells(mask)

        self.sequence.append(sequence)
        self.currImage = self.currImage + 1


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
        if (in_image(cur_image - 1, cell_id)):
            return distance.euclidean(get_cell(i, cell_id).get_centre(), get_cell(cur_image, cell_id).get_centre())
        return 0

    def calculate_total_distance(self, cell_id):
        total = 0
        for i in range(cur_image - 1):
            if (in_image(i, cell_id)):
                total = total + distance.euclidean(get_cell(i, cell_id).get_centre(), get_cell(i + 1, cell_id).get_centre())
        return total

    def calculate_net_distance(self, cell_id):
        for i in range(cur_image):
            if (in_image(i, cell_id)):
                return distance.euclidean(get_cell(i, cell_id).get_centre(), get_cell(cur_image, cell_id).get_centre())
        return 0

    def show_cell_details(self, x, y):
        for cell in self.sequence[cur_image]:
            cell_id = cell.get_id()
            if (cell.contains(x, y) and in_image(cur_image, cell_id)):
                print("Speed: " + str(calculate_speed(cell_id)))
                total_distance = calculate_total_distance(cell_id)
                print("Total Distance: " + str(total_distance))
                net_distance = calculate_net_distance(cell_id)
                print("Net Distance: " + str(net_distance))
                confinement = 0
                if (net_distance != 0):
                    confinement = (total_distance / net_distance)
                print("Confinement Ratio: " + str(confinement))

    def add_cell(self, _id, cnt):
        # TODO: Match cells by characteristic, Don't add a cell we already have
        self.cells.append(Cell(_id, cnt))
        return True

    def count_cells(self, mask):
        _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        sequence = []
        for i, c in enumerate(contours):
            cell = self.add_cell(i, c)
            sequence.append(cell)

        colour = (0, 255, 0)
        
        return len(contours), sequence

    def draw_bounding_box(self,image, cells):
        drawn = image.copy()

        for cell in self.cells:
            colour = (0, 255, 0)
            if cell.dividing:
                colour = (0, 0, 255)

            cv.rectangle(drawn, (cell.get_x(), cell.get_w()), (cell.get_y(), cell.get_h()), colour, 1)
            cv.circle(drawing, center, 1, color, 2)
        
        return drawn