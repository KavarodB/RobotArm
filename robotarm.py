import numpy as np

class Constrained3AxisArm:
    def __init__(self):
        self.column_height = 1.0
        self.hand_length = 0.5

        # Joint states
        self.theta_base = 0.0   # yaw
        self.theta_pitch = 0.0  # vertical angle

        self.last_target = None
        self.predicted_target = None

    def get_end_effector(self):
        pivot = np.array([0, 0, self.column_height])

        dx = self.hand_length * np.cos(self.theta_pitch) * np.cos(self.theta_base)
        dy = self.hand_length * np.cos(self.theta_pitch) * np.sin(self.theta_base)
        dz = self.hand_length * np.sin(self.theta_pitch)
        return pivot + np.array([dx, dy, dz])
    

    def jacobian_ik_update(self, target, dt=0.1):
        pivot = np.array([0, 0, self.column_height])
        end_effector = self.get_end_effector()

        error = target - end_effector

        yaw = self.theta_base
        pitch = self.theta_pitch
        L = self.hand_length

        # ---- Jacobian Matrix ----
        J = np.zeros((3,2))

        J[0,0] = -L*np.cos(pitch)*np.sin(yaw)
        J[1,0] =  L*np.cos(pitch)*np.cos(yaw)
        J[2,0] = 0

        J[0,1] = -L*np.sin(pitch)*np.cos(yaw)
        J[1,1] = -L*np.sin(pitch)*np.sin(yaw)
        J[2,1] =  L*np.cos(pitch)

        # ---- Damped Least Squares Solver (stable IK) ----
        #damping = 0.01

        M = J.T @ J

        dq = np.linalg.solve(M, J.T @ error)

        alpha = 0.95
        dq[0] *= alpha
        dq[1] *= alpha

        # Smooth joint update
        self.theta_base += dq[0] * dt
        self.theta_pitch += dq[1] * dt

        # ---- Forward Kinematics ----
        dx = L * np.cos(self.theta_pitch) * np.cos(self.theta_base)
        dy = L * np.cos(self.theta_pitch) * np.sin(self.theta_base)
        dz = L * np.sin(self.theta_pitch)

        end = pivot + np.array([dx, dy, dz])

        return pivot, end
    
    def track_target(self, target_pos, prediction_gain=1.0):

        target_pos = np.array(target_pos)

        if self.last_target is None:
            self.predicted_target = target_pos
        else:
            # Predict next position using velocity extrapolation
            velocity = target_pos - self.last_target
            self.predicted_target = target_pos + prediction_gain * velocity

        self.last_target = target_pos.copy()

        return self.predicted_target
    
    def ready_to_shoot(self, projectile_speed=0.5, tolerance=0.12):

        if self.predicted_target is None:
            return False

        shooter_pos = self.get_end_effector()

        distance = np.linalg.norm(
            self.predicted_target - shooter_pos
        )

        return distance > tolerance and distance < 6.0
    
    def shoot(self, particle_class, particles_list):

        if not self.ready_to_shoot():
            return False

        shooter_pos = self.get_end_effector()

        particle = particle_class(
            shooter_pos,
            self.predicted_target
        )

        particles_list.append(particle)

        print("RobotArm fired prediction shot")

        return True