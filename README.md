# RobotArm Predictive Shooting Simulation

A Python simulation of a constrained 3-axis robotic arm that tracks moving targets and fires projectiles. Yes, this a Air Defence System simulation.

## Features

- 3-axis arm kinematics with Jacobian-based IK updates
- Kalman-filter-based target state estimation
- Predictive intercept aiming for moving targets
- Projectile simulation with gravity and collision checks
- Visual simulation with Matplotlib animation
- Non-visual benchmark test harness for tracking/shooting error metrics

## Project Structure

- `main.py`: real-time 3D simulation and rendering
- `robotarm.py`: arm model, filtering, prediction, and shooting logic
- `target.py`: target motion models
- `particle.py`: projectile physics and collision checks
- `test.py`: non-visual accuracy benchmark and diagnostics

## Requirements

- Python 3.9+
- `numpy`
- `matplotlib`

Install dependencies:

```bash
pip install numpy matplotlib
```

## Run

Start the visual simulation:

```bash
py main.py
```

Run the non-visual accuracy test:

```bash
py test.py
```

## Tuning Notes

Useful parameters for behavior tuning:

- `Constrained3AxisArm.projectile_speed` in `robotarm.py`
- `Constrained3AxisArm.lead_blend` in `robotarm.py`
- `Constrained3AxisArm.max_target_speed` in `robotarm.py`
- target speed ranges in `main.py` and `test.py`
- collision threshold in `particle.py`

## License

This project is licensed under the MIT License. See `LICENSE` for details.
