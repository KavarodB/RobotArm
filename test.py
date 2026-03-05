import robotarm as robotarm
import target as target
import particle as part
import numpy as np
import random

def test_shooting_accuracy(num_targets=10, frames_per_target=100):
    """
    Test the arm's shooting accuracy without visualization.
    Runs a simulation for multiple targets and collects error metrics.
    """
    dt = 0.03  # simulation time step
    arm = robotarm.Constrained3AxisArm()
    particles = []
    errors = []  # list of (target_pos, predicted_pos, error_distance) tuples

    for target_idx in range(num_targets):
        # Spawn a new target
        new_x = random.uniform(3, 5)
        new_y = random.uniform(-4, 4)
        new_z = random.uniform(2.5, 5)
        new_speed = random.uniform(0.03, 0.1)
        
        target_obj = target.LinearTarget([new_x, new_y, new_z], new_speed)
        arm.reset_filter()  # reset filter for new target
        
        shot_fired = False
        target_errors = []
        
        for frame in range(frames_per_target):
            if target_obj.is_hit:
                break  # target hit, move to next
            
            target_pos = target_obj.update()
            if target_pos is None:
                break  # target out of bounds
            
            # Track and predict
            predicted_pos = arm.track_target(target_pos, dt)
            
            # IK update
            joint, end = arm.jacobian_ik_update(predicted_pos, dt)
            
            # Calculate error: distance between actual target and predicted intercept
            if predicted_pos is not None:
                error_distance = np.linalg.norm(target_pos - predicted_pos)
                target_errors.append(error_distance)
                errors.append((target_pos, predicted_pos, error_distance))
            
            # Shooting logic
            if not shot_fired and arm.ready_to_shoot():
                arm.shoot(part.PredictiveShooterParticle, particles)
                shot_fired = True
            
            # Update particles
            particles = [p for p in particles if p.alive]
            for p in particles:
                p.update()
                if p.check_collision(target_pos):
                    target_obj.hit()
                    p.alive = False
                    break
            
            if not particles:
                shot_fired = False
        
        # After frames, if not hit, record final error
        if not target_obj.is_hit and target_errors:
            avg_error = np.mean(target_errors)
            print(f"Target {target_idx+1}: Not hit, avg error: {avg_error:.3f}")
    
    # Overall statistics
    if errors:
        all_errors = [e[2] for e in errors]
        mean_error = np.mean(all_errors)
        std_error = np.std(all_errors)
        min_error = np.min(all_errors)
        max_error = np.max(all_errors)
        
        print("\nOverall Error Statistics:")
        print(f"Mean error: {mean_error:.3f}")
        print(f"Std deviation: {std_error:.3f}")
        print(f"Min error: {min_error:.3f}")
        print(f"Max error: {max_error:.3f}")
        
        # Error function: could be mean_error, or a weighted function
        # For improvement, perhaps minimize mean_error + std_error
        error_function = mean_error + 0.5 * std_error
        print(f"Composite error function (mean + 0.5*std): {error_function:.3f}")
        
        return error_function, errors
    else:
        print("No errors collected")
        return 0, []

if __name__ == "__main__":
    error_func, error_list = test_shooting_accuracy()
    print(f"\nFinal error function value: {error_func}")