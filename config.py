import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LTRACK_NPZ = os.path.join(HERE, "assets", "L_track_barc.npz")
TRACK_HALF_WIDTH = 0.55 # m
OFFTRACK_SLACK = 0 # m
TRACK_NFE = 1000

MASS = 15.0 # kg
L_F = 0.2 # m, cg to front axle
L_R = 0.3 # m, cg to rear axle
I_Z = 2.8 # kg m^2, yaw inertia
I_X = 2.5 # kg m^2, roll inertia
H = 0.4 # m, cg height
G = 9.81
ROBOT_WIDTH = 0.55 # m, wheel track
CAF = 800 # N/rad, front cornering stiffness
CAR = 750 # N/rad, rear cornering stiffness

V_MIN = 0.3 # m/s
V_MAX = 3.0 # m/s

D_LB = -(TRACK_HALF_WIDTH - ROBOT_WIDTH / 2 + OFFTRACK_SLACK)
D_UB = -D_LB

U0_MIN, U0_MAX = -0.6, 0.6 # m/s^2, longitudinal acceleration
MZ_MIN, MZ_MAX = -75.0, 75.0 # N m, yaw moment
MZ_RATE_MAX = 300.0 # N m/s
BETA_MIN, BETA_MAX = -0.3, 0.3 # rad, sideslip
X2_MIN, X2_MAX = -0.8, 0.8 # rad/s, yaw rate
X3_MIN, X3_MAX = -0.25 * np.pi, 0.25 * np.pi
X4_MIN, X4_MAX = 0.0, 20.0

DYNAMICS_PARAMS = {
    'Caf': CAF, 'Car': CAR,
    'm': MASS, 'Lf': L_F, 'Lr': L_R, 'Iz': I_Z, 'Ix': I_X,
    'g': G, 'h': H,
    'u0_min': U0_MIN, 'u0_max': U0_MAX,
    'mz_min': MZ_MIN, 'mz_max': MZ_MAX, 'mz_rate_max': MZ_RATE_MAX,
    'beta_min': BETA_MIN, 'beta_max': BETA_MAX,
    'x0_min': V_MIN, 'x0_max': V_MAX,
    'x2_min': X2_MIN, 'x2_max': X2_MAX,
    'x3_min': X3_MIN, 'x3_max': X3_MAX,
    'x4_min': X4_MIN, 'x4_max': X4_MAX,
}

# mpc params
DT_MPC = 0.10
HORIZON = 35
AX_MAX = 2.0
M_THETA_MAX = 15.0
PSI_DOT_MAX = float(np.pi / 3)
THETA_MAX = 0.50

W_HEADING = 300.0
W_SPEED = 40.0
W_POSITION = 2000.0
W_BETA = 200.0
W_YAW_RATE = 2000.0
W_BOUNDARY = 16000.0
W_LTR = 50.0
W_ROLL = 5000.0
R_CTRL = np.diag([0.010, 0.001, 0.001])
R_RATE = np.diag([3.0, 1.0, 1.0])

MPC_PARAMS = {'Caf': CAF, 'Car': CAR, 'm': MASS, 'Lf': L_F, 'Lr': L_R,
              'Iz': I_Z, 'Ix': I_X, 'h': H, 'g': G}

N_SUBSTEP = 4

# simulation
ASSETS_DIR = os.path.join(HERE, "assets")
RACELINE_H5 = os.path.join(ASSETS_DIR, "raceline.h5")
PLANT_SUBSTEPS = 10
TRACK_WIDTH = ROBOT_WIDTH
M_PSI_REALIZABLE = 130.0
TRAJECTORY_PLOT = os.path.join(HERE, "track_result.png")
ROLL_PLOT = os.path.join(HERE, "roll_comparison.png")
