
class Cell(object):

    def __init__(self, i, x, y):
        self.id = i
        self.x_min = x
        self.x_max = x
        self.y_min = y
        self.y_max = y
        self.x_velocity = 0
        self.y_velocity = 0
        self.dividing = False
        self.matched = False

    
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
