
class Cell(object):

    def __init__(self, i, x, y):
        self.id = i
        self.x_min = x
        self.x_max = x
        self.y_min = y
        self.y_max = y
        self.x_velocity = 0
        self.y_velocity = 0
    
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