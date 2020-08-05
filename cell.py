class Cell(object):
    def __init__(self, i, cnt):
        self.id = i
        self.cnt = cnt
        self.rect = cv.boundingRect(cnt)
        self.center, _ = cv.minEnclosingCircle(cnt)

        self.x_min = int(self.rect[0])
        self.x_max = int(self.rect[1])
        self.y_min = int(self.x_min+self.rect[2])
        self.y_max = int(self.y_max+self.rect[3])

        self.x_velocity = 0
        self.y_velocity = 0

        self.dividing = False
        self.matched = False
        self.inFrame = True

    def __str__(self):
        return "Cell id: " + str(self.id) + " x range: " + str(self.x_min) + "-" + str(self.x_max) + " y range: " + str(self.y_min) + "-" + str(self.y_max)

    def update_bound(self, x, y):
        if (x < self.x_min):
            self.x_min = x
        if (x > self.x_max):
            self.x_max = x
        if (y < self.y_min):
            self.y_min = y
        if (y > self.y_max):
            self.y_max = y

    def contains(self, x, y):
        if (x < self.x_max and x > self.x_min):
            if (y < self.y_max and y > self.y_min):
                return True
        return False

    def get_dividing(self):
        return self.dividing

    def get_matched(self):
        return self.matched
    
    def get_centre(self):
        return (int((self.x_min + self.x_max)/2),int((self.y_min + self.y_max)/2)) 

    def get_id(self):
        return self.id

    def get_x_min(self):
        return self.x_min

    def get_x_max(self):
        return self.x_max

    def get_y_min(self):
        return self.y_min

    def get_y_max(self):
        return self.y_max

    def get_x_velocity(self):
        return self.x_velocity

    def get_y_velocity(self):
        return self.y_velocity
    
    def set_id(self, new_id):
        self.id = new_id

    def set_x_min(self, new_x):
        self.x_min = new_x

    def set_x_max(self, new_x):
        self.x_max = new_x

    def set_y_min(self, new_y):
        self.y_min = new_y

    def set_y_max(self, new_y):
        self.y_max = new_y

    def set_x_velocity(self, new_x_velocity):
        self.x_velocity = new_x_velocity

    def set_y_velocity(self, new_y_velocity):
        self.y_velocity = new_y_velocity

    def set_dividing(self):
        self.dividing = True

    def set_matched(self):
        self.matched = True

class CellManager(object):

    def __init__(self, dataset):
        self.cells = []
        self.currImage = 0

    def count_cell_divisions(self, cells):
        count = 0
        for cell in cells:
            if (cell.get_dividing()):
                count = count + 1
        return count


    def in_image(self, image_no, cell_id):
        if (image_no < 0 or image_no >= len(sequence)):
            return False
        for cell in sequence[image_no]:
            if (cell.get_id() == cell_id):
                return True
        return False

    def get_cell(self, image_no, cell_id):
        for cell in sequence[image_no]:
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
        for cell in sequence[cur_image]:
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

    def on_click(self, event, x, y, p1, p2):
        if event == cv2.EVENT_LBUTTONDOWN:
            show_cell_details(x, y)

    def add_cell(self, _id, x, y, cnt):
        # TODO: Match cells by characteristic, Don't add a cell we already have
        self.cells.append(Cell(_id, x, y, cnt))

    def count_cells(self, mask):
        _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        for i, c in enumerate(contours):
            self.add_cell(i, c)

        colour = (0, 255, 0)
        
        return len(contours)

    def draw_bounding_box(self,image, cells):
        drawn = image.copy()

        for cell in self.cells:
            colour = (0, 255, 0)
            if cell.dividing:
                colour = (0, 0, 255)

            cv2.rectangle(drawn, (cell.get_x_min(), cell.get_y_min()), (cell.get_x_max(), cell.get_y_max()), colour, 1)
            cv.circle(drawing, center, 1, color, 2)
        
        return drawn