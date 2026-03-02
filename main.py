import robotarm as robotarm
import target as target
import particle as part
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

FRAMES = 30

def run_simulation():
    
    arm = robotarm.Constrained3AxisArm()
    target_obj = target.CircularTarget()
    particles = []
    shot_fired = False
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(-4,4)
    ax.set_ylim(-4,4)
    ax.set_zlim(0,5)

    column_line, = ax.plot([], [], [], lw=6)
    hand_line, = ax.plot([], [], [], lw=3)
    target_dot, = ax.plot([], [], [], 'ro')
    particle_dots, = ax.plot([], [], [], 'bo', markersize=3)


    def ready_toShoot(arm, target_pos, tolerance=0.12):
        shooter_pos = arm.get_end_effector()

        distance = np.linalg.norm(target_pos - shooter_pos)

        # Check if target is within reachable engagement zone
        return distance > tolerance and distance < 3.0


    def update(frame):
        nonlocal particles
        nonlocal shot_fired

        target_pos = target_obj.update()

        target_pos = target_obj.update()
        arm.track_target(target_pos)

        if not shot_fired:
            arm.shoot(part.PredictiveShooterParticle, particles)
            shot_fired = True

        joint, end = arm.jacobian_ik_update(target_pos)
        
        px = []
        py = []
        pz = []

        if not any(p.alive for p in particles):
            shot_fired = False

        # ---- Particle Simulation ----
        for particle in particles:
            if not particle.alive:
                continue

            particle.update()
            particle.check_collision(target_pos)
            particle.check_out_of_bounds()

            px.append(particle.position[0])
            py.append(particle.position[1])
            pz.append(particle.position[2])


        particle_dots.set_data(px, py)
        particle_dots.set_3d_properties(pz)
        
        # Column (always upright)
        column_line.set_data([0,0], [0,0])
        column_line.set_3d_properties([0, arm.column_height])

        # Hand
        hand_line.set_data([joint[0], end[0]],
                           [joint[1], end[1]])
        hand_line.set_3d_properties([joint[2], end[2]])

        target_dot.set_data([target_pos[0]], [target_pos[1]])
        target_dot.set_3d_properties([target_pos[2]])

        return column_line, hand_line, target_dot

    ani = FuncAnimation(fig, update,
                        frames=1000,
                        interval=FRAMES,
                        blit=False)

    plt.title("Constrained 3-Axis Arm Tracking")
    plt.show()


if __name__ == "__main__":
    run_simulation()