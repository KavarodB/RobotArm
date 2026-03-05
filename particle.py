import numpy as np

class PredictiveShooterParticle:
    def __init__(self, shooter_pos, target_pos,
                 projectile_speed=0.55):

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
        
        # remember previous position for segment-based collision checks
        self.prev_position = self.position.copy()

        # Apply gravity
        self.velocity += self.gravity

        # Move particle
        self.position += self.velocity

        return self.position
    
    def check_collision(self, target, threshold=0.15):
        if not self.alive:
            return False

        # check current position first
        if np.linalg.norm(self.position - target) < threshold:
            self.alive = False
            print("Collision detected with target!")
            return True

        # also check along the segment from previous to current
        if hasattr(self, 'prev_position'):
            a = self.prev_position
            b = self.position
            p = target
            ab = b - a
            if np.dot(ab, ab) > 1e-9:
                t = np.dot(p - a, ab) / np.dot(ab, ab)
                t = np.clip(t, 0.0, 1.0)
                closest = a + ab * t
                if np.linalg.norm(closest - p) < threshold:
                    self.alive = False
                    print("Collision detected with target! (segment)")
                    return True

        return False
    
    def check_out_of_bounds(self, bounds):
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        x, y, z = self.position
        if not (xmin <= x <= xmax and
            ymin <= y <= ymax and
            zmin <= z <= zmax):

            self.alive = False
            return True

        return False