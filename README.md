# 2R Robot Arm Inverse Dynamics — Learned vs. Physics-Structured Models

Compares three neural-network approaches to learning the inverse dynamics of a planar 2R robot arm — predicting joint torques `τ` from joint angles `θ`, velocities `θ̇`, and accelerations `θ̈` — against ground truth from an analytical robot model.

## Models

- **Model 1 — Structured Coefficient**: predicts 11 physical coefficients (mass matrix, Coriolis, gravity terms) from `θ`, then combines them via the manipulator equation `τ = M(θ)θ̈ + C(θ,θ̇)θ̇ + G(θ)`.
- **Model 2 — DeLaN-Inspired**: learns a Cholesky-factorized (SPD) mass matrix and a potential energy function; Coriolis and gravity are derived via finite-difference Christoffel symbols and the potential's gradient.
- **Model 3 — Direct Baseline**: plain feed-forward network mapping `[θ, θ̇, θ̈] → τ` with no physical structure.

## Data

10,000 samples of `(θ, θ̇, θ̈)` drawn randomly, with torques computed via `SerialManipulator.inverse_dynamics(...)` on an example 2R arm (`serial_manipulator_mixed_joints.py`). Same train/val split used across all models.

## Files

```
├── code.ipynb                          # Data generation, all 3 models, training, evaluation
└── serial_manipulator_mixed_joints.py  # Robot kinematics/dynamics (forward/inverse kinematics, mass matrix, Coriolis, dynamics, RK4 sim)
```

## Setup & Usage

```bash
pip install tensorflow numpy
```

The notebook hardcodes a local `sys.path`/`os.chdir` — update those paths to wherever `serial_manipulator_mixed_joints.py` lives on your machine, then run all cells. Each model trains for 300 epochs (Adam, MSE loss, batch size 64), followed by torque-accuracy comparison and mass-matrix/coefficient sanity checks.

## Known Issue

The evaluation step calls `robot.coriolis_vector(q, qdot)`, but `SerialManipulator` only exposes `coriolis_matrix(q, q_dot)` — swap in `robot.coriolis_matrix(q, qdot) @ qdot` to fix.

## License

Add a license of your choice (e.g. MIT).
