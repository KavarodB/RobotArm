import robotarm as robotarm
import target as target
import particle as part
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random

FRAMES = 30

def run_simulation():
    
    arm = robotarm.Constrained3AxisArm()
    target_obj = target.LinearTarget()
    particles = []
    shot_fired = False
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(-4,4)
    ax.set_ylim(-4,4)
    ax.set_zlim(0,5)

    BOUNDS = [-4,4,-4,4,0,5]

    column_line, = ax.plot([], [], [], lw=6)
    hand_line, = ax.plot([], [], [], lw=3)
    target_dot, = ax.plot([], [], [], 'ro')
    particle_dots, = ax.plot([], [], [], 'bo', markersize=3)

    def update(frame):
        nonlocal particles
        nonlocal shot_fired
        
        # 1. Handle "Disappeared" state
        if target_obj.is_hit:
            new_x = random.uniform(3, 5)
            new_y = random.uniform(-4, 4)
            new_z = random.uniform(2.5,5)
            new_speed = random.uniform(0.03,0.1)

            target_obj.spawn([new_x, new_y, new_z], speed=new_speed)
            shot_fired = False # Reset shooter for the new target
            print(f"New target spawned at: {new_x:.2f}, {new_y:.2f}")

            return

        target_pos = target_obj.update()
        # 2. Prediction & Tracking
        # Use the predicted position for the IK, not the current raw target_pos
        predicted_pos = arm.track_target(target_pos)
        joint, end = arm.jacobian_ik_update(target_pos)

        # 3. Shooting Logic
        if not shot_fired:
            # The arm shoots toward its predicted_target
            arm.shoot(part.PredictiveShooterParticle, particles)
            shot_fired = True

        px, py, pz = [], [], []

        # 4. Particle & Collision Logic
        # Filter out dead particles for the next frame
        particles = [p for p in particles if p.alive]
        
        if not particles:
            shot_fired = False

        else:
            for particle in particles:
                particle.update()
                
                # Check if particle hits the active target
                if particle.check_collision(target_pos):
                    target_obj.hit()
                    particle.alive = False
                    print("🎯 Target Neutralized!")
                
                particle.check_out_of_bounds(BOUNDS)

                if particle.alive:
                    px.append(particle.position[0])
                    py.append(particle.position[1])
                    pz.append(particle.position[2])

        # 5. Rendering Updates
        particle_dots.set_data(px, py)
        particle_dots.set_3d_properties(pz)
        
        column_line.set_data([0, 0], [0, 0])
        column_line.set_3d_properties([0, arm.column_height])

        hand_line.set_data([joint[0], end[0]], [joint[1], end[1]])
        hand_line.set_3d_properties([joint[2], end[2]])

        target_dot.set_data([target_pos[0]], [target_pos[1]])
        target_dot.set_3d_properties([target_pos[2]])

        return column_line, hand_line, target_dot, particle_dots

    ani = FuncAnimation(fig, update,
                        frames=1000,
                        interval=FRAMES,
                        blit=False)

    plt.title("Constrained 3-Axis Arm Tracking")
    plt.show()


if __name__ == "__main__":
    run_simulation()