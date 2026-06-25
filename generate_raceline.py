import os
import h5py
import numpy as np
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.environ import (
    ConcreteModel, Constraint, ConstraintList, Objective, Param, SolverFactory,
    TransformationFactory, Var, cos, minimize, sin, value,
)
from scipy.interpolate import CubicSpline

from config import DYNAMICS_PARAMS, TRACK_NFE, TRACK_HALF_WIDTH, D_LB, D_UB, LTRACK_NPZ

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets", "raceline.h5")
N_PTS = 200

class LTrackMap:
    def __init__(self, npz_path, width=None):
        data = np.load(npz_path)
        segs = data['cl_segs'] 
        track_w = float(data['track_width'])
        self.width = width if width is not None else track_w / 2.0

        self._segs = segs
        self._s0 = np.concatenate([[0.0], np.cumsum(segs[:, 0])])
        track_length = float(self._s0[-1])

        kp = np.zeros((len(segs) + 1, 3))
        for i, (length, radius) in enumerate(segs):
            x0, y0, psi0 = kp[i]
            if radius == 0.0:
                kp[i+1] = [x0 + length*np.cos(psi0), y0 + length*np.sin(psi0), psi0]
            else:
                kappa = 1.0 / radius
                psi1 = psi0 + kappa * length
                kp[i+1] = [x0 + (np.sin(psi1) - np.sin(psi0)) / kappa,
                           y0 + (np.cos(psi0) - np.cos(psi1)) / kappa, psi1]
        self._kp = kp

        N = 2000
        s_sample = np.linspace(0.0, track_length, N, endpoint=False)
        xy_sample = np.array([self._xy_at(s) for s in s_sample])
        s_closed = np.append(s_sample, track_length)
        xy_closed = np.vstack([xy_sample, xy_sample[0]])
        self.cs = CubicSpline(s_closed, xy_closed, bc_type='periodic')
        self.TrackLength = track_length

    def _xy_at(self, s):
        s = s % self._s0[-1]
        idx = min(int(np.searchsorted(self._s0, s, side='right')) - 1, len(self._segs) - 1)
        ds = s - self._s0[idx]
        x0, y0, psi0 = self._kp[idx]
        radius = self._segs[idx, 1]
        if radius == 0.0:
            return np.array([x0 + ds * np.cos(psi0), y0 + ds * np.sin(psi0)])
        kappa = 1.0 / radius
        psi = psi0 + kappa * ds
        return np.array([x0 + (np.sin(psi) - np.sin(psi0)) / kappa,
                         y0 + (np.cos(psi0) - np.cos(psi)) / kappa])

    def getCurvaturePrev(self, s):
        tl = self._s0[-1]
        while s > tl:
            s -= tl
        idx = min(int(np.searchsorted(self._s0, s, side='right')) - 1, len(self._segs) - 1)
        radius = self._segs[idx, 1]
        return 0.0 if radius == 0.0 else 1.0 / radius

    def getGlobalPosition(self, s, ey, epsi, return_tangent=False):
        xy = self.cs(s)
        ds = self.cs(s, 1)
        n = ds / np.linalg.norm(ds)
        n_t = np.array([n[1], -n[0]])
        xy = xy - ey * n_t
        x, y = xy[0], xy[1]
        psi_track = np.arctan2(n[1], n[0])
        psi = epsi + psi_track
        if not return_tangent:
            return x, y
        return x, y, psi, n[0], n[1]

def model(track_length, params):
    m = ConcreteModel()
    m.sf = Param(initialize=track_length)
    m.s = ContinuousSet(bounds=(0, m.sf))

    m.u0 = Var(m.s, bounds=(params['u0_min'], params['u0_max']), initialize=0)
    m.u1 = Var(m.s, bounds=(params['mz_min'], params['mz_max']), initialize=0)

    m.x0 = Var(m.s, bounds=(params['x0_min'], params['x0_max']), initialize=1.0)
    m.x1 = Var(m.s, bounds=(params['beta_min'], params['beta_max']), initialize=0)
    m.x2 = Var(m.s, bounds=(params['x2_min'], params['x2_max']), initialize=0)
    m.x3 = Var(m.s, bounds=(params['x3_min'], params['x3_max']), initialize=0)
    m.x4 = Var(m.s, bounds=(params['x4_min'], params['x4_max']), initialize=0)
    m.x5 = Var(m.s, bounds=(D_LB, D_UB), initialize=0)

    m.dx0ds = DerivativeVar(m.x0, wrt=m.s)
    m.dx1ds = DerivativeVar(m.x1, wrt=m.s)
    m.dx2ds = DerivativeVar(m.x2, wrt=m.s)
    m.dx3ds = DerivativeVar(m.x3, wrt=m.s)
    m.dx4ds = DerivativeVar(m.x4, wrt=m.s)
    m.dx5ds = DerivativeVar(m.x5, wrt=m.s)
    m.du1ds = DerivativeVar(m.u1, wrt=m.s)

    m.obj = Objective(expr=m.x4[m.sf], sense=minimize)
    return m

def vehicle_model(m, params, track_map, eps=1e-6):
    mass, lf, lr, Iz = params["m"], params["Lf"], params["Lr"], params["Iz"]
    Caf, Car = params["Caf"], params["Car"]

    def _G(m, s):
        cur = track_map.getCurvaturePrev(s)
        return (1 - cur * m.x5[s]) / (eps + m.x0[s] * cos(m.x1[s] + m.x3[s]))

    def _x0dot(m, s):
        return m.dx0ds[s] == m.u0[s] * _G(m, s)
    m.x0dot = Constraint(m.s, rule=_x0dot)

    def _x1dot(m, s):
        v_eps = (m.x0[s] ** 2 + 0.2) ** 0.5
        dbeta = (m.x1[s] * (-(Caf + Car) / (mass * v_eps))
                 + m.x2[s] * ((-Caf * lf + Car * lr) / (mass * v_eps ** 2) - 1))
        return m.dx1ds[s] == dbeta * _G(m, s)
    m.x1dot = Constraint(m.s, rule=_x1dot)

    def _x2dot(m, s):
        v_eps = (m.x0[s] ** 2 + 0.2) ** 0.5
        ddpsi = (m.x1[s] * ((-Caf * lf + Car * lr) / Iz)
                 - m.x2[s] * (Caf * lf ** 2 + Car * lr ** 2) / (Iz * v_eps)
                 + m.u1[s] / Iz)
        return m.dx2ds[s] == ddpsi * _G(m, s)
    m.x2dot = Constraint(m.s, rule=_x2dot)

    def _x3dot(m, s):
        cur = track_map.getCurvaturePrev(s)
        return m.dx3ds[s] == m.x2[s] * _G(m, s) - cur
    m.x3dot = Constraint(m.s, rule=_x3dot)

    def _x4dot(m, s):
        return m.dx4ds[s] == _G(m, s)
    m.x4dot = Constraint(m.s, rule=_x4dot)

    def _x5dot(m, s):
        return m.dx5ds[s] == m.x0[s] * sin(m.x1[s] + m.x3[s]) * _G(m, s)
    m.x5dot = Constraint(m.s, rule=_x5dot)

def mz_rate_model(m, track_map, params, eps=1e-6):
    def _G(m, s):
        cur = track_map.getCurvaturePrev(s)
        return (1 - cur * m.x5[s]) / (eps + m.x0[s] * cos(m.x1[s] + m.x3[s]))

    def _u1dotmax(m, s):
        return m.du1ds[s] <= params['mz_rate_max'] * _G(m, s)
    m.i1dot_max = Constraint(m.s, rule=_u1dotmax)

    def _u1dotmin(m, s):
        return m.du1ds[s] >= -params['mz_rate_max'] * _G(m, s)
    m.i1dot_min = Constraint(m.s, rule=_u1dotmin)

def boundary_conditions(m, track_length):
    def _init(m):
        yield m.x0[0] == m.x0[track_length]
        yield m.x1[0] == m.x1[track_length]
        yield m.x2[0] == m.x2[track_length]
        yield m.x3[0] == m.x3[track_length]
        yield m.x4[0] == 0
        yield m.x5[0] == m.x5[track_length]
    m.init_conditions = ConstraintList(rule=_init)

def build_raceline_xy(m, track_map):
    svec = sorted(m.s.value)
    L = track_map.TrackLength
    x_out, y_out, theta_out, v_out = [], [], [], []
    for sj in svec:
        d = float(value(m.x5[sj]))
        heading_err = float(value(m.x3[sj]))
        gx, gy, gpsi, _, _ = track_map.getGlobalPosition(sj % L, d, heading_err, return_tangent=True)
        x_out.append(gx)
        y_out.append(gy)
        theta_out.append(gpsi)
        v_out.append(float(value(m.x0[sj])))
    return x_out, y_out, theta_out, v_out

def smooth_raceline(x, y, theta, v, n_pts=N_PTS):
    x_arr = np.array(x)
    y_arr = np.array(y)
    diffs = np.hypot(np.diff(x_arr, append=x_arr[0]), np.diff(y_arr, append=y_arr[0]))
    ss = np.concatenate([[0.0], np.cumsum(diffs[:-1])])
    L = float(ss[-1]) + float(diffs[-1])

    ss_c = np.append(ss, L)
    xy_c = np.column_stack([np.append(x_arr, x_arr[0]), np.append(y_arr, y_arr[0])])
    v_c = np.append(np.array(v), v[0])

    keep = np.concatenate([[True], np.diff(ss_c) > 1e-9])
    cs = CubicSpline(ss_c[keep], xy_c[keep], bc_type='periodic')

    ss_new = np.linspace(0.0, L, n_pts, endpoint=False)
    xy_new = cs(ss_new)
    dxy = cs(ss_new, 1)
    theta_new = np.arctan2(dxy[:, 1], dxy[:, 0])
    v_new = np.interp(ss_new, ss_c[keep], v_c[keep])
    return xy_new[:, 0], xy_new[:, 1], theta_new, v_new


def main():
    params = DYNAMICS_PARAMS
    track_map = LTrackMap(LTRACK_NPZ, width=TRACK_HALF_WIDTH)

    m = model(track_map.TrackLength, params)
    vehicle_model(m, params, track_map)
    mz_rate_model(m, track_map, params)
    boundary_conditions(m, track_map.TrackLength)
    TransformationFactory("dae.finite_difference").apply_to(m, nfe=TRACK_NFE)

    solver = SolverFactory("ipopt")
    if solver is None or not solver.available():
        raise RuntimeError("ipopt not available - install it (e.g. conda install -c conda-forge ipopt).")
    solver.solve(m, tee=True)

    x, y, theta, v = build_raceline_xy(m, track_map)
    x, y, theta, v = smooth_raceline(x, y, theta, v)

    with h5py.File(OUT, "w") as f:
        f.create_dataset("x", data=np.asarray(x))
        f.create_dataset("y", data=np.asarray(y))
        f.create_dataset("psi", data=np.asarray(theta))
        f.create_dataset("v", data=np.asarray(v))
    print(f"raceline ({len(x)} pts) saved -> {OUT}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sb = np.linspace(0.0, track_map.TrackLength, 600, endpoint=False)
    inner = np.array([track_map.getGlobalPosition(s, -TRACK_HALF_WIDTH, 0) for s in sb])
    outer = np.array([track_map.getGlobalPosition(s, TRACK_HALF_WIDTH, 0) for s in sb])
    plt.figure(figsize=(10, 8))
    plt.plot(inner[:, 0], inner[:, 1], "k", lw=1)
    plt.plot(outer[:, 0], outer[:, 1], "k", lw=1)
    sc = plt.scatter(x, y, c=v, cmap="viridis", s=20, label="raceline (color=v)")
    plt.colorbar(sc, label="v [m/s]")
    plt.legend()
    plt.axis("equal")
    out_png = os.path.splitext(OUT)[0] + "_plot.png"
    plt.savefig(out_png, dpi=120)
    print(f"plot saved -> {out_png}")


if __name__ == "__main__":
    main()
