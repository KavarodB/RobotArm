import numpy as np

class InfinityTarget:
    def __init__(self, scale=2.0):
        self.t = 0
        self.scale = scale
        self.speed = 0.05  # affects object velocity

    def update(self):
        self.t += self.speed

        x = self.scale * np.sin(self.t)
        y = self.scale * np.sin(self.t) * np.cos(self.t)
        z = 2 + 0.7 * np.cos(self.t)

        return np.array([x, y, z])
    
class CircularTarget:
    def __init__(self, radius=2.0, z_height=2.0, speed=0.01):
        self.radius = radius
        self.z = z_height
        self.t = 0
        self.speed = speed

    def update(self):
        self.t += self.speed

        x = self.radius * np.cos(self.t)
        y = self.radius * np.sin(self.t)
        z = self.z

        return np.array([x, y, z])