import cv2 as cv
import numpy as np

from scipy.spatial import distance

from utilis import *

import PhC
import Flou
class Cell(object):
    def __init__(self, i, cnt):
        self.id = i
        self.contours = cnt
        self.rect = cv.boundingRect(cnt)
        self.centre, _ = cv.minEnclosingCircle(cnt)

        self.x_velocity = 0
        self.y_velocity = 0

        self.dividing = False
        self.matched = False
        self.inFrame = True

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

    def __init__(self):
        self.cells = []
        self.currImage = 0
        self.sequence = []

    def count_cell_divisions(self, cells):
        count = 0
        for cell in cells:
            if (cell.is_dividing()):
                count = count + 1
        return count

    def processImage(self, filename):
        img = cv.imread(filename)

        #processes images to segmented and thresholded cells
        #replace with better segmentation algorithm
        mask, img = PhC.preprocess(img)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h = hMaxima(gray)
        hmax = nFoldDilation(gray, h)

        #counts labelled cells, measures bounding boxes and stores in list
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        pred_count, sequence = self.count_cells(mask)

        self.sequence.append(sequence)
        drawn = self.draw_bounding_box(img)
        
        cv.imshow("Bounding Box", drawn)
        cv.waitKey(0)
        cv.destroyAllWindows()

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
        if self.in_image(self.currImage - 1, cell_id):
            return distance.euclidean(get_cell(i, cell_id).get_centre(), get_cell(self.currImage, cell_id).get_centre())
        return 0

    def calculate_total_distance(self, cell_id):
        total = 0
        for i in range(self.currImage - 1):
            if (in_image(i, cell_id)):
                total = total + distance.euclidean(get_cell(i, cell_id).get_centre(), get_cell(i + 1, cell_id).get_centre())
        return total

    def calculate_net_distance(self, cell_id):
        for i in range(self.currImage):
            if self.in_image(i, cell_id):
                cell1 = get_cell(i, cell_id)
                cell2 = get_cell(self.currImage, cell_id)
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
        # TODO: Match cells by characteristic, Don't add a cell we already have
        return Cell(_id, cnt)

    def count_cells(self, mask):
        _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        sequence = []
        for i, c in enumerate(contours):
            cell = self.add_cell(i, c)
            sequence.append(cell)

        colour = (0, 255, 0)
        
        return len(contours), sequence

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