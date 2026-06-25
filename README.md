# go2w-roll-control-mpc
Supplementary code and video for the paper: Racing a Wheeled Quadruped: Active Load Transfer Mitigation via Model Predictive Control
by Marla Eisman, Brian Lam, Samuel Sonnino, and Francesco Borrelli from the MPC Lab at UC Berkeley and Politecnico di Milano, Italy. Check out the video of the experiment below:

https://github.com/user-attachments/assets/1c9027e6-6691-4b0d-b7d6-2f30f8803af6

This repository contains the high-level active-roll-control MPC from the paper and a closed-loop
simulation of it on the dynamic-bicycle model.

## files

```
config.py - robot, track, MPC, and simulation parameters
generate_raceline.py - offline minimum-time raceline optimization
mpc.py - dynamic-bicycle MPC, outputs [a_x, M_psi, M_theta]
simulate.py - closed-loop MPC simulation + plots
assets/
  L_track_barc.npz - track centerline segments
  raceline.h5 - precomputed raceline (x, y, psi, v)
```

## install

```bash
pip install -r requirements.txt
conda install -c conda-forge ipopt
```

## run

```bash
python generate_raceline.py # optional, regenerates assets/raceline.h5
python simulate.py # closed-loop MPC, writes track_result.png + roll_comparison.png
python simulate.py --show # also open the figures interactively
```