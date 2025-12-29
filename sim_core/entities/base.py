import numpy as np

class Entity:
    def __init__(self, uid, team):
        self.uid = uid
        self.team = team # 0: Red, 1: Blue
        self.is_active = True # False means dead or removed
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.radius = 5.0 # 碰撞体积半径

    def update(self, dt, all_entities):
        raise NotImplementedError