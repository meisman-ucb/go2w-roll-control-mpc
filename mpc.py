import numpy as np
import casadi as ca
from scipy.spatial import KDTree

import config as C

# state: x y psi beta dpsi v theta dtheta
# control: a_x, M_psi (yaw moment), M_theta (roll moment)

def _load_track(path):
    import h5py
    with h5py.File(path, 'r') as h5f:
        x = np.asarray(h5f['x'], dtype=float)
        y = np.asarray(h5f['y'], dtype=float)
        theta = np.unwrap(np.asarray(h5f['psi'], dtype=float))
        v = np.asarray(h5f['v'], dtype=float)
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    ss = np.cumsum(np.hypot(dx, dy))
    return {'x': x, 'y': y, 'theta': theta, 'v': v, 's': ss, 'L': float(ss[-1])}


class _BicycleMPC:

    def __init__(self, track, v_max=C.V_MAX, v_min=C.V_MIN):
        self.track = track
        self.v_max = v_max
        self.v_min = v_min
        self.p = C.MPC_PARAMS
        self.kdtree = KDTree(np.column_stack((track['x'], track['y'])))
        self.nx, self.nu, self.T = 8, 3, C.HORIZON
        self._x_warm = None
        self._u_warm = None
        self._build_dynamics()
        self._build()

    def _build_dynamics(self):
        p = self.p
        Caf, Car, m, Lf, Lr, Iz = p['Caf'], p['Car'], p['m'], p['Lf'], p['Lr'], p['Iz']
        Ix, h, g = p['Ix'], p['h'], p['g']
        z = ca.SX.sym('z', self.nx)
        u = ca.SX.sym('u', self.nu)
        psi, beta, dpsi, v, theta, dtheta = z[2], z[3], z[4], z[5], z[6], z[7]
        a_x, Mz, M_roll = u[0], u[1], u[2]
        v_eps = ca.sqrt(v**2 + 0.2)
        dbeta = beta * (-(Caf + Car) / (m * v_eps)) \
            + dpsi * ((-Caf * Lf + Car * Lr) / (m * v_eps**2) - 1)
        ddpsi = beta * ((-Caf * Lf + Car * Lr) / Iz) \
            - dpsi * ((Caf * Lf**2 + Car * Lr**2) / (Iz * v_eps)) + Mz / Iz
        a_y = v_eps * (dbeta + dpsi)
        ddtheta = (-m * g * h * ca.sin(theta)
                   - m * a_y * h * ca.cos(theta) + M_roll) / Ix
        dz = ca.vertcat(v * ca.cos(psi + beta), v * ca.sin(psi + beta),
                        dpsi, dbeta, ddpsi, a_x, dtheta, ddtheta)
        fc = ca.Function('fc', [z, u], [dz])

        dt_sub = C.DT_MPC / C.N_SUBSTEP
        zk = z
        for _ in range(C.N_SUBSTEP):
            k1 = fc(zk, u)
            k2 = fc(zk + 0.5 * dt_sub * k1, u)
            k3 = fc(zk + 0.5 * dt_sub * k2, u)
            k4 = fc(zk + dt_sub * k3, u)
            zk = zk + (dt_sub / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.fd = ca.Function('fd', [z, u], [zk])

    def _build(self):
        opti = ca.Opti()
        opti.solver('ipopt',
            {"expand": True, "print_time": 0, "verbose": False, "error_on_fail": 0},
            {"max_iter": 40, "print_level": 0, "tol": 1e-3,
             "acceptable_tol": 5e-2, "acceptable_iter": 3,
             "warm_start_init_point": "yes"})

        x = opti.variable(self.nx, self.T + 1)
        u = opti.variable(self.nu, self.T)
        x0 = opti.parameter(self.nx)
        xr = opti.parameter(self.T + 1)
        yr = opti.parameter(self.T + 1)
        pr = opti.parameter(self.T + 1)
        vr = opti.parameter(self.T + 1)

        opti.subject_to(x[:, 0] == x0)
        for k in range(self.T):
            opti.subject_to(x[:, k+1] == self.fd(x[:, k], u[:, k]))

        cost = 0
        R = ca.DM(C.R_CTRL)
        Rd = ca.DM(C.R_RATE)
        for k in range(self.T + 1):
            ex = x[0, k] - xr[k]
            ey = x[1, k] - yr[k]
            cost += C.W_POSITION * (ex**2 + ey**2)
            cost += C.W_HEADING * (1 - ca.cos(x[2, k] - pr[k]))
            cost += C.W_SPEED * (x[5, k] - vr[k])**2

            cte = -ca.sin(pr[k]) * ex + ca.cos(pr[k]) * ey
            cost += C.W_BOUNDARY * ca.fmax(0, ca.fabs(cte) - C.TRACK_HALF_WIDTH)**2

            beta_k = x[3, k]
            cost += C.W_BETA * (ca.fmax(0, beta_k - C.BETA_MAX)**2 +
                                ca.fmax(0, -beta_k - C.BETA_MAX)**2)

            dpsi_k = x[4, k]
            cost += C.W_YAW_RATE * (ca.fmax(0, dpsi_k - C.PSI_DOT_MAX)**2 +
                                    ca.fmax(0, -dpsi_k - C.PSI_DOT_MAX)**2)

            theta_k = x[6, k]
            cost += C.W_LTR * (x[5, k] * dpsi_k + C.G * theta_k)**2
            cost += C.W_ROLL * (ca.fmax(0, theta_k - C.THETA_MAX)**2 +
                                ca.fmax(0, -theta_k - C.THETA_MAX)**2)

            opti.subject_to(x[5, k] >= self.v_min)
            opti.subject_to(x[5, k] <= self.v_max)

            if k < self.T:
                opti.subject_to(u[0, k] >= -C.AX_MAX)
                opti.subject_to(u[0, k] <= C.AX_MAX)
                opti.subject_to(u[2, k] >= -C.M_THETA_MAX)
                opti.subject_to(u[2, k] <= C.M_THETA_MAX)
                cost += u[:, k].T @ R @ u[:, k]
                if k >= 1:
                    du = u[:, k] - u[:, k-1]
                    cost += du.T @ Rd @ du

        opti.minimize(cost)

        self._opti = opti
        self._x = x
        self._u = u
        self._x0 = x0
        self._xr = xr
        self._yr = yr
        self._pr = pr
        self._vr = vr

    def _arc_length(self, x_pos, y_pos):
        _, idx = self.kdtree.query([float(x_pos), float(y_pos)])
        t = self.track
        dx = float(x_pos) - t['x'][idx]
        dy = float(y_pos) - t['y'][idx]
        ds = dx * np.cos(t['theta'][idx]) + dy * np.sin(t['theta'][idx])
        return (t['s'][idx] + ds) % t['L']

    def solve(self, x0_arr):
        x0_safe = np.array(x0_arr, dtype=float).copy()
        x0_safe[5] = max(x0_safe[5], self.v_min)

        s = self._arc_length(x0_safe[0], x0_safe[1])
        t = self.track
        L = t['L']
        xrv = np.zeros(self.T + 1)
        yrv = np.zeros(self.T + 1)
        prv = np.zeros(self.T + 1)
        vrv = np.zeros(self.T + 1)
        for k in range(self.T + 1):
            sw = s % L
            xrv[k] = np.interp(sw, t['s'], t['x'])
            yrv[k] = np.interp(sw, t['s'], t['y'])
            prv[k] = np.interp(sw, t['s'], t['theta'])
            base_v = float(np.interp(sw, t['s'], t['v']))
            vrv[k] = float(np.clip(base_v, self.v_min, self.v_max))
            s += vrv[k] * C.DT_MPC

        prv = np.unwrap(prv)
        x0_safe[2] = prv[0] + ((x0_safe[2] - prv[0] + np.pi) % (2 * np.pi) - np.pi)
        self._opti.set_value(self._x0, x0_safe)

        self._opti.set_value(self._xr, xrv)
        self._opti.set_value(self._yr, yrv)
        self._opti.set_value(self._pr, prv)
        self._opti.set_value(self._vr, vrv)

        if self._x_warm is None:
            self._opti.set_initial(self._x, x0_safe.reshape((self.nx, 1)) @ np.ones((1, self.T + 1)))
            self._opti.set_initial(self._u, np.zeros((self.nu, self.T)))
        else:
            self._opti.set_initial(self._x, self._x_warm)
            self._opti.set_initial(self._u, self._u_warm)

        try:
            sol = self._opti.solve()
            x_opt = sol.value(self._x)
            u_opt = sol.value(self._u)
        except Exception:
            x_opt = self._opti.debug.value(self._x)
            u_opt = self._opti.debug.value(self._u)

        if not (np.all(np.isfinite(x_opt)) and np.all(np.isfinite(u_opt))):
            self._x_warm = None
            return None, None, False

        self._x_warm = np.hstack((x_opt[:, 1:], x_opt[:, -1:]))
        self._u_warm = np.hstack((u_opt[:, 1:], u_opt[:, -1:]))
        return x_opt, u_opt, True


class MomentMPC:

    def __init__(self, raceline=C.RACELINE_H5, v_max=C.V_MAX):
        self.track = _load_track(raceline)
        self.mpc = _BicycleMPC(self.track, v_max=v_max)
        self.track_length = self.track['L']

    def arc_length(self, x, y):
        return self.mpc._arc_length(x, y)

    def control(self, z, tilt=True):
        x_opt, u_opt, ok = self.mpc.solve(z)
        if not ok:
            return np.zeros(3), 0.0
        a_x = float(np.clip(u_opt[0, 0], -C.AX_MAX, C.AX_MAX))
        m_psi = float(u_opt[1, 0])
        m_theta = float(np.clip(u_opt[2, 0], -C.M_THETA_MAX, C.M_THETA_MAX)) if tilt else 0.0
        theta_des = float(np.clip(x_opt[6, 1], -C.THETA_MAX, C.THETA_MAX))
        return np.array([a_x, m_psi, m_theta], dtype=float), theta_des
