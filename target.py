import numpy as np

class InfinityTarget:
    def __init__(self, scale=2.0):
        self.t = 0
        self.scale = scale
        self.speed = 0.05  # affects object velocity
        self.is_hit = False

    def update(self):
        self.t += self.speed

        x = self.scale * np.sin(self.t)
        y = self.scale * np.sin(self.t) * np.cos(self.t)
        z = 2 + 0.7 * np.cos(self.t)

        return np.array([x, y, z])
    
    def hit(self):
        """Call this when a projectile collision is detected"""
        self.is_hit = True
    
class CircularTarget:
    def __init__(self, radius=3.0, z_height=2.0, speed=0.05):
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
    
    def hit(self):
        """Call this when a projectile collision is detected"""
        self.is_hit = True

class LinearTarget:
    def __init__(self, start_pos=[5.0, 5.0, 2.0], speed=0.065):
        self.spawn(start_pos, speed)

    def spawn(self, pos, speed):
        self.start_pos = np.array(pos, dtype=float)
        self.pos = self.start_pos.copy()
        self.is_hit = False
        direction = np.array([-1.0, -1.0, -0.1]) 
        self.velocity = (direction / np.linalg.norm(direction)) * speed

    def update(self):
        if self.is_hit: return None
        self.pos += self.velocity
        if self.pos[2] < 0.1 or np.linalg.norm(self.pos) > 11.0:
            self.is_hit = True # Mark for respawn if it goes out of bounds
        return self.pos

    def hit(self):
        self.is_hit = True