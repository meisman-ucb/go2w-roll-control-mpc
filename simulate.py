from __future__ import annotations
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from mpc import MomentMPC
import config as C

P = C.MPC_PARAMS


def deriv(z, u):
    psi, beta, dpsi, v, theta = z[2], z[3], z[4], z[5], z[6]
    a_x, mz, m_roll = u
    Caf, Car, m, Lf, Lr = P['Caf'], P['Car'], P['m'], P['Lf'], P['Lr']
    Iz, Ix, h, g = P['Iz'], P['Ix'], P['h'], P['g']
    v_eps = np.sqrt(v * v + 0.2)
    dbeta = beta * (-(Caf + Car) / (m * v_eps)) \
        + dpsi * ((-Caf * Lf + Car * Lr) / (m * v_eps ** 2) - 1)
    ddpsi = beta * ((-Caf * Lf + Car * Lr) / Iz) \
        - dpsi * ((Caf * Lf ** 2 + Car * Lr ** 2) / (Iz * v_eps)) + mz / Iz
    a_y = v_eps * (dbeta + dpsi)
    ddtheta = (-m * g * h * np.sin(theta) - m * a_y * h * np.cos(theta) + m_roll) / Ix
    dz = np.array([v * np.cos(psi + beta), v * np.sin(psi + beta),
                   dpsi, dbeta, ddpsi, a_x, z[7], ddtheta])
    return dz, a_y


def step_plant(z, u, dt, nsub):
    hstep = dt / nsub
    a_y = 0.0
    for _ in range(nsub):
        k1, a_y = deriv(z, u)
        k2, _ = deriv(z + 0.5 * hstep * k1, u)
        k3, _ = deriv(z + 0.5 * hstep * k2, u)
        k4, _ = deriv(z + hstep * k3, u)
        z = z + (hstep / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z, a_y


def ltr(a_y, theta):
    return (2.0 * P['h'] / C.TRACK_WIDTH) * ((a_y / P['g']) * np.cos(theta) + np.sin(theta))


def theta_star(v, dpsi):
    return -np.arctan(v * dpsi / P['g'])


def initial_state(track):
    kappa = np.gradient(track['theta'], track['s'])
    v0 = float(np.clip(track['v'][0], C.V_MIN, C.V_MAX))
    return np.array([track['x'][0], track['y'][0], track['theta'][0],
                     0.0, float(kappa[0]) * v0, v0, 0.0, 0.0])


def run(planner):
    track = planner.track
    tree = KDTree(np.column_stack((track['x'], track['y'])))
    z = initial_state(track)
    L = planner.track_length
    dt = C.DT_MPC

    keys = ['x', 'y', 'v', 'theta', 'theta_star', 'a_y', 'cte', 'a_x', 'm_psi', 'm_theta', 's', 't']
    log = {k: [] for k in keys}
    last_u = np.zeros(3)
    dist = 0.0
    t = 0.0

    for _ in range(int(2.5 * L / C.V_MIN / dt)):
        u, _ = planner.control(z, tilt=True)
        if abs(u[1]) > C.M_PSI_REALIZABLE or not np.all(np.isfinite(u)):
            u = last_u
        last_u = u
        z, a_y = step_plant(z, u, dt, C.PLANT_SUBSTEPS)

        x, y, dpsi, v, theta = z[0], z[1], z[4], z[5], z[6]
        _, idx = tree.query([x, y])
        tpsi = track['theta'][idx]
        cte = -np.sin(tpsi) * (x - track['x'][idx]) + np.cos(tpsi) * (y - track['y'][idx])

        log['x'].append(x)
        log['y'].append(y)
        log['v'].append(v)
        log['theta'].append(theta)
        log['theta_star'].append(theta_star(v, dpsi))
        log['a_y'].append(a_y)
        log['cte'].append(cte)
        log['a_x'].append(u[0])
        log['m_psi'].append(u[1])
        log['m_theta'].append(u[2])
        log['s'].append(dist)
        log['t'].append(t)

        dist += abs(v) * dt
        t += dt
        if dist >= L:
            break

    out = {k: np.asarray(val) for k, val in log.items()}
    out['ltr_on'] = ltr(out['a_y'], out['theta'])
    out['ltr_off'] = ltr(out['a_y'], 0.0)
    return out


def track_edges(npz_path, ds=0.02):
    d = np.load(npz_path)
    segs = d["cl_segs"]
    half = float(d["track_width"]) / 2.0
    cx, cy, cpsi = [], [], []
    x0 = y0 = psi0 = 0.0
    for length, radius in segs:
        n = max(2, int(length / ds))
        for s in np.linspace(0.0, length, n, endpoint=False):
            if radius == 0.0:
                cx.append(x0 + s * np.cos(psi0)); cy.append(y0 + s * np.sin(psi0)); cpsi.append(psi0)
            else:
                kappa = 1.0 / radius
                psi = psi0 + kappa * s
                cx.append(x0 + (np.sin(psi) - np.sin(psi0)) / kappa)
                cy.append(y0 + (np.cos(psi0) - np.cos(psi)) / kappa); cpsi.append(psi)
        if radius == 0.0:
            x0 += length * np.cos(psi0); y0 += length * np.sin(psi0)
        else:
            kappa = 1.0 / radius
            psi1 = psi0 + kappa * length
            x0 += (np.sin(psi1) - np.sin(psi0)) / kappa
            y0 += (np.cos(psi0) - np.cos(psi1)) / kappa
            psi0 = psi1
    cx, cy, cpsi = np.array(cx), np.array(cy), np.array(cpsi)
    nx, ny = np.sin(cpsi), -np.cos(cpsi)
    close = lambda a: np.vstack([a, a[0]])
    left = close(np.column_stack((cx - half * nx, cy - half * ny)))
    right = close(np.column_stack((cx + half * nx, cy + half * ny)))
    return left, right


def plot_trajectory(d, path):
    left, right = track_edges(C.LTRACK_NPZ)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    for a in ax:
        a.plot(left[:, 0], left[:, 1], 'k', lw=1.2)
        a.plot(right[:, 0], right[:, 1], 'k', lw=1.2)
        a.set_aspect('equal')
        a.set_xlabel('x [m]')
        a.set_ylabel('y [m]')

    s0 = ax[0].scatter(d['x'], d['y'], c=d['v'], cmap='viridis', s=16)
    fig.colorbar(s0, ax=ax[0], label='speed [m/s]', shrink=0.8)
    ax[0].set_title('velocity on optimal trajectory')

    s1 = ax[1].scatter(d['x'], d['y'], c=np.degrees(d['theta']),
                       cmap='coolwarm', s=16, vmin=-15, vmax=15)
    fig.colorbar(s1, ax=ax[1], label='roll angle [deg]', shrink=0.8)
    ax[1].set_title('active roll on optimal trajectory')

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f'saved {path}')


def plot_comparison(d, path):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    ax[0, 0].plot(d['s'], d['v'], color='tab:green')
    ax[0, 0].set_title('speed around track')
    ax[0, 0].set_xlabel('distance [m]')
    ax[0, 0].set_ylabel('v [m/s]')

    ax[0, 1].plot(d['s'], np.degrees(d['theta']), color='tab:red', label=r'commanded roll $\theta$')
    ax[0, 1].plot(d['s'], np.degrees(d['theta_star']), '--', color='k', lw=1, label=r'optimal $\theta^*$')
    ax[0, 1].set_title('active roll around track')
    ax[0, 1].set_xlabel('distance [m]')
    ax[0, 1].set_ylabel(r'$\theta$ [deg]')
    ax[0, 1].legend()

    ax[1, 0].plot(d['s'], d['ltr_off'], color='tab:gray', label='roll off')
    ax[1, 0].plot(d['s'], d['ltr_on'], color='tab:blue', label='roll on')
    ax[1, 0].axhline(1.0, color='r', ls=':', lw=1)
    ax[1, 0].axhline(-1.0, color='r', ls=':', lw=1)
    ax[1, 0].set_title('LTR around track')
    ax[1, 0].set_xlabel('distance [m]')
    ax[1, 0].set_ylabel('LTR')
    ax[1, 0].legend()

    ax[1, 1].scatter(np.abs(d['a_y']), np.abs(d['ltr_off']), s=12, color='tab:gray', label='roll off')
    ax[1, 1].scatter(np.abs(d['a_y']), np.abs(d['ltr_on']), s=12, color='tab:blue', label='roll on')
    ax[1, 1].set_title('LTR vs lateral acceleration')
    ax[1, 1].set_xlabel(r'$|a_y|$ [m/s$^2$]')
    ax[1, 1].set_ylabel('|LTR|')
    ax[1, 1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f'saved {path}')


def report(d):
    on, off = np.abs(d['ltr_on']), np.abs(d['ltr_off'])
    print(f'\n  lap time          {d["t"][-1]:.2f} s')
    print(f'  mean / peak speed {np.mean(d["v"]):.2f} / {np.max(d["v"]):.2f} m/s')
    print(f'  peak |a_y|        {np.max(np.abs(d["a_y"])):.2f} m/s^2')
    print(f'  mean |CTE|        {np.mean(np.abs(d["cte"])):.3f} m')
    print(f'  peak body bank    {np.degrees(np.max(np.abs(d["theta"]))):.1f} deg')
    print(f'\n  {"":16s} {"roll off":>9s} {"roll on":>9s}')
    print(f'  {"mean |LTR|":16s} {np.mean(off):>9.3f} {np.mean(on):>9.3f}   ({100*(np.mean(on)-np.mean(off))/np.mean(off):.0f}%)')
    print(f'  {"peak |LTR|":16s} {np.max(off):>9.3f} {np.max(on):>9.3f}   ({100*(np.max(on)-np.max(off))/np.max(off):.0f}%)')
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()
    if not args.show:
        matplotlib.use('Agg')

    planner = MomentMPC()
    print('running closed-loop MPC...')
    d = run(planner)
    report(d)
    plot_trajectory(d, C.TRAJECTORY_PLOT)
    plot_comparison(d, C.ROLL_PLOT)

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
