import numpy as np

class Constrained3AxisArm:
    def __init__(self):
        self.column_height = 0.7
        self.hand_length = 0.5

        # Joint states
        self.theta_base = 0.0   # yaw
        self.theta_pitch = 0.0  # vertical angle

        self.last_target = None
        self.predicted_target = None

        self.kf_state = np.zeros(6)
        self.kf_covariance = np.eye(6) * 1.0
        self.last_time = None

    def update_filter(self, measurement, dt):
        # Prediction Step
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt  # Transition matrix
        
        self.kf_state = F @ self.kf_state
        Q = np.eye(6) * 0.1 # Process noise
        self.kf_covariance = F @ self.kf_covariance @ F.T + Q

        # Update Step (Correction)
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        
        z = measurement
        y = z - H @ self.kf_state # Residual
        S = H @ self.kf_covariance @ H.T + np.eye(3) * 0.01 # Measurement noise
        K = self.kf_covariance @ H.T @ np.linalg.inv(S)
        
        self.kf_state = self.kf_state + K @ y
        self.kf_covariance = (np.eye(6) - K @ H) @ self.kf_covariance

    def get_intercept_point(self, projectile_speed):
        """
        Calculates the point in space where the projectile and target meet.
        Uses the iterative approach for Time-of-Flight.
        """
        target_pos = self.kf_state[0:3]
        target_vel = self.kf_state[3:6]
        shooter_pos = self.get_end_effector()
        
        t = 0
        for _ in range(3): # 3 iterations is usually enough for convergence
            dist = np.linalg.norm(target_pos + target_vel * t - shooter_pos)
            t = dist / projectile_speed
            
        return target_pos + target_vel * t

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
    
    def track_target(self, target_pos, dt=0.1):
        self.update_filter(np.array(target_pos), dt)
        
        # Predict the actual intercept point based on projectile speed
        # Assume projectile_speed = 5.0 for this example
        self.predicted_target = self.get_intercept_point(projectile_speed=5.0)
        return self.predicted_target
    
    def ready_to_shoot(self, tolerance=0.12):

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

        return True