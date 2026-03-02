import numpy as np

class PredictiveShooterParticle:
    def __init__(self, shooter_pos, target_pos,
                 projectile_speed=0.3):

        self.position = np.array(shooter_pos, dtype=float)

        direction = target_pos - shooter_pos

        if np.linalg.norm(direction) < 1e-12:
            direction = np.array([1,0,0])
        else:
            direction = direction / np.linalg.norm(direction)

        self.velocity = direction * projectile_speed

        self.gravity = np.array([0,0,-0.01])
        self.alive = True

    def update(self):
        if not self.alive:
            return self.position

        # Apply gravity
        self.velocity += self.gravity

        # Move particle
        self.position += self.velocity

        return self.position
    
    def check_collision(self, target, threshold=0.15):
        if not self.alive:
            return False

        if np.linalg.norm(self.position - target) < threshold:
            self.alive = False
            print("Collision detected with target!")

            return True

        return False
    
    def check_out_of_bounds(self, bounds=(-4, 4,-4, 4,0, 5)):
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        x, y, z = self.position
        if not (xmin <= x <= xmax and
            ymin <= y <= ymax and
            zmin <= z <= zmax):

            self.alive = False
            return True

        return False