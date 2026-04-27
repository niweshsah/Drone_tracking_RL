import argparse
import csv
import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as pe
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

from environment import DroneTrackingEnv
from agent import TD3Agent, TD3Config


# ==========================================
# 0. GLOBAL STYLE CONFIGURATION
# ==========================================
# Clean, publication-quality light theme
DARK_BG      = "#FFFFFF"
PANEL_BG     = "#F8F9FA"
GRID_COLOR   = "#E0E0E0"
TEXT_COLOR   = "#212121"
RL_COLOR     = "#1976D2"   # Deep Blue  – RL agent
PD_COLOR     = "#D32F2F"   # Deep Red   – PD baseline
TGT_COLOR    = "#388E3C"   # Deep Green – Target
ACCENT       = "#F57C00"   # Orange     – Highlights

RL_CMAP  = LinearSegmentedColormap.from_list("rl_cmap",  ["#BBDEFB", RL_COLOR])
PD_CMAP  = LinearSegmentedColormap.from_list("pd_cmap",  ["#FFCDD2", PD_COLOR])
ERR_CMAP = LinearSegmentedColormap.from_list("err_cmap", ["#C8E6C9", "#FFF9C4", "#FFCDD2"])

def apply_global_style():
    matplotlib.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    GRID_COLOR,
        "axes.labelcolor":   TEXT_COLOR,
        "axes.titlecolor":   TEXT_COLOR,
        "axes.grid":         True,
        "axes.titlesize":    12,
        "axes.labelsize":    10,
        "grid.color":        GRID_COLOR,
        "grid.linewidth":    0.6,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.facecolor":  DARK_BG,
        "legend.edgecolor":  GRID_COLOR,
        "legend.labelcolor": TEXT_COLOR,
        "legend.fontsize":   8,
        "text.color":        TEXT_COLOR,
        "lines.linewidth":   1.8,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.facecolor": DARK_BG,
        "font.family":       "DejaVu Sans",
    })

apply_global_style()


# ==========================================
# 1. BASELINE CONTROLLER
# ==========================================
class BaselinePDController:
    """
    A simple PD controller to serve as a baseline comparison against the RL agent.
    Uses bounding box center (position error) and current velocity.
    """
    def __init__(self, kp: float = 1.5, kd: float = 0.3):
        self.kp = kp
        self.kd = kd

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        err_x, err_y = obs[0], obs[1]
        vel_x, vel_y = obs[4], obs[5]
        action_x = self.kp * err_x - self.kd * vel_x
        action_y = self.kp * err_y - self.kd * vel_y
        return np.clip([action_x, action_y], -1.0, 1.0)


# ==========================================
# 2. ADVANCED METRICS CALCULATION
# ==========================================
def compute_advanced_metrics(
    drone_pos: np.ndarray, target_pos: np.ndarray,
    drone_vel: np.ndarray, target_vel: np.ndarray,
    actions: np.ndarray, dt: float
) -> Dict[str, float]:
    """Calculates an extensive list of kinematic and control metrics."""

    pos_errors = np.linalg.norm(target_pos[:, :2] - drone_pos[:, :2], axis=1)
    rmse_error      = np.sqrt(np.mean(pos_errors**2))
    max_error       = np.max(pos_errors)
    mean_error      = np.mean(pos_errors)
    median_error    = np.median(pos_errors)
    p95_error       = np.percentile(pos_errors, 95)
    std_error       = np.std(pos_errors)

    # Per-axis errors
    x_errors = np.abs(target_pos[:, 0] - drone_pos[:, 0])
    y_errors = np.abs(target_pos[:, 1] - drone_pos[:, 1])

    # Velocity metrics
    drone_speeds  = np.linalg.norm(drone_vel[:, :2], axis=1)
    target_speeds = np.linalg.norm(target_vel[:, :2], axis=1)
    speed_dev     = np.abs(drone_speeds - target_speeds)
    mean_speed_dev = np.mean(speed_dev)

    # Heading / angle error
    drone_dir  = np.arctan2(drone_vel[:, 1],  drone_vel[:, 0])
    target_dir = np.arctan2(target_vel[:, 1], target_vel[:, 0])
    heading_err = np.abs(np.arctan2(np.sin(target_dir - drone_dir),
                                    np.cos(target_dir - drone_dir)))
    mean_heading_err = np.degrees(np.mean(heading_err))

    # Acceleration 
    if len(drone_vel) > 1:
        accel = np.diff(drone_vel[:, :2], axis=0) / dt
        accel_mag = np.linalg.norm(accel, axis=1)
        mean_accel = np.mean(accel_mag)
        max_accel  = np.max(accel_mag)
    else:
        mean_accel = max_accel = 0.0

    # Control effort & smoothness
    action_mag   = np.linalg.norm(actions, axis=1)
    ctrl_effort  = np.mean(action_mag)
    ctrl_peak    = np.max(action_mag)
    if len(actions) > 1:
        act_delta  = np.diff(actions, axis=0)
        act_jitter = np.mean(np.linalg.norm(act_delta, axis=1))
    else:
        act_jitter = 0.0

    # Settling & overshoot – fraction of time within 0.5 m
    within_05m = float(np.mean(pos_errors < 0.5)) * 100
    within_1m  = float(np.mean(pos_errors < 1.0)) * 100

    # Path efficiency: actual distance / shortest distance
    d_trav = float(np.sum(np.linalg.norm(np.diff(drone_pos[:, :2], axis=0), axis=1)))
    d_ideal= float(np.sum(np.linalg.norm(np.diff(target_pos[:, :2], axis=0), axis=1)))
    path_eff = (d_ideal / d_trav * 100) if d_trav > 0 else 100.0

    return {
        "RMSE (m)":            float(rmse_error),
        "Max Error (m)":       float(max_error),
        "Mean Error (m)":      float(mean_error),
        "Median Error (m)":    float(median_error),
        "P95 Error (m)":       float(p95_error),
        "Std Error (m)":       float(std_error),
        "X Mean Error (m)":    float(np.mean(x_errors)),
        "Y Mean Error (m)":    float(np.mean(y_errors)),
        "Speed Dev (m/s)":     float(mean_speed_dev),
        "Heading Err (deg)":   float(mean_heading_err),
        "Mean Accel (m/s²)":   float(mean_accel),
        "Max Accel (m/s²)":    float(max_accel),
        "Control Effort":      float(ctrl_effort),
        "Peak Control":        float(ctrl_peak),
        "Action Jitter":       float(act_jitter),
        "Time within 0.5m (%)": float(within_05m),
        "Time within 1.0m (%)": float(within_1m),
        "Path Efficiency (%)": float(path_eff),
    }


# ==========================================
# 3. HELPER: GRADIENT LINE COLLECTION
# ==========================================
def _gradient_line(ax, x, y, cmap, linewidth=2.0, alpha=1.0):
    """Draw a line whose color transitions along its length."""
    pts   = np.array([x, y]).T.reshape(-1, 1, 2)
    segs  = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm  = Normalize(0, len(segs))
    lc    = LineCollection(segs, cmap=cmap, norm=norm,
                           linewidth=linewidth, alpha=alpha)
    lc.set_array(np.arange(len(segs)))
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def _label_patch(color, label):
    return mpatches.Patch(color=color, label=label)


# ==========================================
# 4. INDIVIDUAL PLOT SUITES
# ==========================================

def plot_2d_trajectory(ax, rl_h, pd_h, traj_name):
    """Beautiful 2-D spatial trajectory with gradient coloring."""
    tx, ty = rl_h["t_pos"][:, 0], rl_h["t_pos"][:, 1]
    ax.plot(tx, ty, color=TGT_COLOR, lw=2.5, ls="--", label="Target", zorder=5)
    ax.scatter(tx[0],  ty[0],  c=TGT_COLOR, s=80, zorder=10, marker="o")
    ax.scatter(tx[-1], ty[-1], c=TGT_COLOR, s=80, zorder=10, marker="s")

    _gradient_line(ax, rl_h["d_pos"][:, 0], rl_h["d_pos"][:, 1], RL_CMAP, 2.2)
    _gradient_line(ax, pd_h["d_pos"][:, 0], pd_h["d_pos"][:, 1], PD_CMAP, 2.2)

    ax.legend(handles=[
        _label_patch(TGT_COLOR, "Target"),
        _label_patch(RL_COLOR,  "RL (start→end: light→dark)"),
        _label_patch(PD_COLOR,  "PD (start→end: light→dark)"),
    ])
    ax.set_title(f"2-D Spatial Trajectory  [{traj_name}]")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="datalim")


def plot_error_time(ax, rl_h, pd_h, dt):
    t_rl = np.arange(len(rl_h["d_pos"])) * dt
    t_pd = np.arange(len(pd_h["d_pos"])) * dt
    rl_e = np.linalg.norm(rl_h["t_pos"][:, :2] - rl_h["d_pos"][:, :2], axis=1)
    pd_e = np.linalg.norm(pd_h["t_pos"][:, :2] - pd_h["d_pos"][:, :2], axis=1)
    sm_rl = gaussian_filter1d(rl_e, sigma=4)
    sm_pd = gaussian_filter1d(pd_e, sigma=4)

    ax.fill_between(t_rl, rl_e, alpha=0.15, color=RL_COLOR)
    ax.fill_between(t_pd, pd_e, alpha=0.15, color=PD_COLOR)
    ax.plot(t_rl, rl_e,  color=RL_COLOR,  alpha=0.35, lw=0.8)
    ax.plot(t_pd, pd_e,  color=PD_COLOR,  alpha=0.35, lw=0.8)
    ax.plot(t_rl, sm_rl, color=RL_COLOR,  lw=2.5, label=f"RL (RMSE={np.sqrt(np.mean(rl_e**2)):.2f}m)")
    ax.plot(t_pd, sm_pd, color=PD_COLOR,  lw=2.5, label=f"PD (RMSE={np.sqrt(np.mean(pd_e**2)):.2f}m)")
    ax.axhline(0.5, color=ACCENT, ls=":", lw=1.2, label="0.5 m threshold")
    ax.set_title("Tracking Error over Time")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Euclidean Error (m)")
    ax.legend()


def plot_per_axis_error(ax_x, ax_y, rl_h, pd_h, dt):
    t_rl = np.arange(len(rl_h["d_pos"])) * dt
    t_pd = np.arange(len(pd_h["d_pos"])) * dt
    for ax, axis_i, label in [(ax_x, 0, "X"), (ax_y, 1, "Y")]:
        rl_e = np.abs(rl_h["t_pos"][:, axis_i] - rl_h["d_pos"][:, axis_i])
        pd_e = np.abs(pd_h["t_pos"][:, axis_i] - pd_h["d_pos"][:, axis_i])
        ax.fill_between(t_rl, gaussian_filter1d(rl_e, 4), alpha=0.2, color=RL_COLOR)
        ax.fill_between(t_pd, gaussian_filter1d(pd_e, 4), alpha=0.2, color=PD_COLOR)
        ax.plot(t_rl, gaussian_filter1d(rl_e, 4), color=RL_COLOR, lw=2, label="RL")
        ax.plot(t_pd, gaussian_filter1d(pd_e, 4), color=PD_COLOR, lw=2, label="PD")
        ax.set_title(f"{label}-Axis Error over Time")
        ax.set_xlabel("Time (s)"); ax.set_ylabel(f"|Δ{label}| (m)")
        ax.legend()


def plot_speed_profile(ax, rl_h, pd_h, dt):
    t_rl = np.arange(len(rl_h["d_vel"])) * dt
    t_pd = np.arange(len(pd_h["d_vel"])) * dt
    tgt_spd = np.linalg.norm(rl_h["t_vel"][:, :2], axis=1)
    rl_spd  = np.linalg.norm(rl_h["d_vel"][:, :2], axis=1)
    pd_spd  = np.linalg.norm(pd_h["d_vel"][:, :2], axis=1)
    ax.fill_between(t_rl, rl_spd, tgt_spd, alpha=0.12, color=RL_COLOR)
    ax.fill_between(t_pd, pd_spd, tgt_spd, alpha=0.12, color=PD_COLOR)
    ax.plot(t_rl, tgt_spd, color=TGT_COLOR, lw=2.5, ls="--", label="Target Speed")
    ax.plot(t_rl, rl_spd,  color=RL_COLOR,  lw=2,   label="RL Speed")
    ax.plot(t_pd, pd_spd,  color=PD_COLOR,  lw=2,   label="PD Speed")
    ax.set_title("Speed Profile")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (m/s)")
    ax.legend()


def plot_control_effort(ax, rl_h, pd_h, dt):
    t_rl = np.arange(len(rl_h["act"])) * dt
    t_pd = np.arange(len(pd_h["act"])) * dt
    rl_n = np.linalg.norm(rl_h["act"], axis=1)
    pd_n = np.linalg.norm(pd_h["act"], axis=1)
    ax.fill_between(t_rl, gaussian_filter1d(rl_n, 3), alpha=0.2, color=RL_COLOR)
    ax.fill_between(t_pd, gaussian_filter1d(pd_n, 3), alpha=0.2, color=PD_COLOR)
    ax.plot(t_rl, gaussian_filter1d(rl_n, 3), color=RL_COLOR, lw=2, label=f"RL (mean={np.mean(rl_n):.3f})")
    ax.plot(t_pd, gaussian_filter1d(pd_n, 3), color=PD_COLOR, lw=2, label=f"PD (mean={np.mean(pd_n):.3f})")
    ax.set_title("Control Effort (Action Norm)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("|Action|")
    ax.legend()


def plot_action_channels(ax_x, ax_y, rl_h, pd_h, dt):
    t_rl = np.arange(len(rl_h["act"])) * dt
    t_pd = np.arange(len(pd_h["act"])) * dt
    for ax, ch, label in [(ax_x, 0, "Action X"), (ax_y, 1, "Action Y")]:
        ax.plot(t_rl, gaussian_filter1d(rl_h["act"][:, ch], 3), color=RL_COLOR, lw=2, label="RL")
        ax.plot(t_pd, gaussian_filter1d(pd_h["act"][:, ch], 3), color=PD_COLOR, lw=2, label="PD")
        ax.axhline( 1.0, color=ACCENT, ls=":", lw=1.0)
        ax.axhline(-1.0, color=ACCENT, ls=":", lw=1.0)
        ax.set_title(f"{label} Channel over Time")
        ax.set_xlabel("Time (s)"); ax.set_ylabel(label)
        ax.set_ylim(-1.2, 1.2)
        ax.legend()


def plot_error_distribution(ax, rl_h, pd_h):
    rl_e = np.linalg.norm(rl_h["t_pos"][:, :2] - rl_h["d_pos"][:, :2], axis=1)
    pd_e = np.linalg.norm(pd_h["t_pos"][:, :2] - pd_h["d_pos"][:, :2], axis=1)
    bins = np.linspace(0, max(rl_e.max(), pd_e.max()) * 1.05, 40)
    ax.hist(rl_e, bins=bins, color=RL_COLOR, alpha=0.5, label="RL", density=True)
    ax.hist(pd_e, bins=bins, color=PD_COLOR, alpha=0.5, label="PD", density=True)
    # KDE overlay
    for e, c in [(rl_e, RL_COLOR), (pd_e, PD_COLOR)]:
        xs = np.linspace(0, e.max() * 1.1, 300)
        try:
            kde = gaussian_kde(e, bw_method=0.3)
            ax.plot(xs, kde(xs), color=c, lw=2.5)
        except Exception:
            pass
    for e, c, lbl in [(rl_e, RL_COLOR, "RL"), (pd_e, PD_COLOR, "PD")]:
        ax.axvline(np.mean(e), color=c, ls="--", lw=1.5)
    ax.set_title("Error Distribution (KDE + Histogram)")
    ax.set_xlabel("Euclidean Error (m)"); ax.set_ylabel("Density")
    ax.legend()


def plot_phase_space(ax, h, color, cmap, label):
    """Position error vs velocity for one controller."""
    err = np.linalg.norm(h["t_pos"][:, :2] - h["d_pos"][:, :2], axis=1)
    spd = np.linalg.norm(h["d_vel"][:, :2], axis=1)
    sc  = ax.scatter(err, spd, c=np.arange(len(err)), cmap=cmap,
                     s=8, alpha=0.7, linewidths=0)
    ax.set_xlabel("Position Error (m)")
    ax.set_ylabel("Drone Speed (m/s)")
    ax.set_title(f"Phase Space  [{label}]")
    return sc


def plot_action_scatter(ax, rl_h, pd_h):
    rl_a, pd_a = rl_h["act"], pd_h["act"]
    ax.scatter(rl_a[:, 0], rl_a[:, 1], c=RL_COLOR, s=6, alpha=0.4, label="RL")
    ax.scatter(pd_a[:, 0], pd_a[:, 1], c=PD_COLOR, s=6, alpha=0.4, label="PD")
    circle = plt.Circle((0, 0), 1.0, color=ACCENT, fill=False, ls=":", lw=1.5)
    ax.add_patch(circle)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_title("Action-Space Scatter")
    ax.set_xlabel("Action X"); ax.set_ylabel("Action Y")
    ax.legend()


def plot_heading_error(ax, rl_h, pd_h, dt):
    for h, color, label in [(rl_h, RL_COLOR, "RL"), (pd_h, PD_COLOR, "PD")]:
        if len(h["d_vel"]) > 1:
            dd = np.arctan2(h["d_vel"][:, 1], h["d_vel"][:, 0])
            td = np.arctan2(h["t_vel"][:, 1], h["t_vel"][:, 0])
            herr = np.degrees(np.abs(np.arctan2(np.sin(td - dd), np.cos(td - dd))))
            t    = np.arange(len(herr)) * dt
            ax.fill_between(t, gaussian_filter1d(herr, 4), alpha=0.15, color=color)
            ax.plot(t, gaussian_filter1d(herr, 4), color=color, lw=2,
                    label=f"{label} (mean={np.mean(herr):.1f}°)")
    ax.set_title("Heading Error over Time")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Heading Error (°)")
    ax.legend()


def plot_cumulative_error(ax, rl_h, pd_h, dt):
    rl_e = np.linalg.norm(rl_h["t_pos"][:, :2] - rl_h["d_pos"][:, :2], axis=1)
    pd_e = np.linalg.norm(pd_h["t_pos"][:, :2] - pd_h["d_pos"][:, :2], axis=1)
    t_rl = np.arange(len(rl_e)) * dt
    t_pd = np.arange(len(pd_e)) * dt
    ax.plot(t_rl, np.cumsum(rl_e) * dt, color=RL_COLOR, lw=2.5, label=f"RL (total={np.sum(rl_e)*dt:.2f} m·s)")
    ax.plot(t_pd, np.cumsum(pd_e) * dt, color=PD_COLOR, lw=2.5, label=f"PD (total={np.sum(pd_e)*dt:.2f} m·s)")
    ax.set_title("Cumulative Integrated Error")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("∫ Error dt  (m·s)")
    ax.legend()


def plot_error_heatmap(ax, h, color, title):
    """2-D spatial error density map."""
    ex = h["t_pos"][:, 0] - h["d_pos"][:, 0]
    ey = h["t_pos"][:, 1] - h["d_pos"][:, 1]
    try:
        xy   = np.vstack([ex, ey])
        kde  = gaussian_kde(xy, bw_method=0.4)
        xr   = np.linspace(ex.min()-0.3, ex.max()+0.3, 120)
        yr   = np.linspace(ey.min()-0.3, ey.max()+0.3, 120)
        XX, YY = np.meshgrid(xr, yr)
        ZZ   = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        ax.contourf(XX, YY, ZZ, levels=20, cmap=ERR_CMAP)
        ax.contour(XX, YY, ZZ, levels=8, colors=color, linewidths=0.6, alpha=0.5)
    except Exception:
        ax.scatter(ex, ey, c=color, s=4, alpha=0.3)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("ΔX (m)"); ax.set_ylabel("ΔY (m)")
    ax.set_aspect("equal", adjustable="datalim")


def plot_3d_trajectory(ax, rl_h, pd_h):
    """3-D trajectory (x, y, time)."""
    n_rl = len(rl_h["d_pos"])
    n_pd = len(pd_h["d_pos"])
    t_rl = np.arange(n_rl)
    t_pd = np.arange(n_pd)
    ax.plot(rl_h["t_pos"][:, 0], rl_h["t_pos"][:, 1], t_rl, color=TGT_COLOR, lw=2, ls="--", label="Target")
    ax.plot(rl_h["d_pos"][:, 0], rl_h["d_pos"][:, 1], t_rl, color=RL_COLOR,  lw=2, label="RL")
    ax.plot(pd_h["d_pos"][:, 0], pd_h["d_pos"][:, 1], t_pd, color=PD_COLOR,  lw=2, label="PD")
    ax.set_title("3-D Trajectory (X–Y–Time)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Step")
    ax.legend()
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.grid(True)


def plot_polar_action(ax, rl_h, pd_h):
    """Polar histogram of action directions."""
    for h, color, label in [(rl_h, RL_COLOR, "RL"), (pd_h, PD_COLOR, "PD")]:
        angles = np.arctan2(h["act"][:, 1], h["act"][:, 0])
        bins   = np.linspace(-np.pi, np.pi, 25)
        counts, _ = np.histogram(angles, bins=bins)
        theta  = 0.5 * (bins[:-1] + bins[1:])
        width  = bins[1] - bins[0]
        ax.bar(theta, counts / counts.max(), width=width * 0.9,
               color=color, alpha=0.55, label=label, bottom=0.0)
    ax.set_title("Action Direction Distribution")
    ax.legend(loc="upper right")


def plot_rolling_rmse(ax, rl_h, pd_h, dt, window=30):
    """Rolling window RMSE."""
    for h, color, label in [(rl_h, RL_COLOR, "RL"), (pd_h, PD_COLOR, "PD")]:
        e   = np.linalg.norm(h["t_pos"][:, :2] - h["d_pos"][:, :2], axis=1)
        rmse_roll = [np.sqrt(np.mean(e[max(0, i-window):i+1]**2)) for i in range(len(e))]
        t   = np.arange(len(e)) * dt
        ax.plot(t, rmse_roll, color=color, lw=2, label=label)
    ax.set_title(f"Rolling RMSE (window={window} steps)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Rolling RMSE (m)")
    ax.legend()


def plot_metrics_radar(ax, rl_m, pd_m, metrics_keys):
    """Radar / spider chart comparing key normalised metrics."""
    labels = [k.split("(")[0].strip() for k in metrics_keys]
    n      = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    # Normalise: 0 = best, 1 = worst  (for error/jitter metrics)
    vals_rl, vals_pd = [], []
    for key in metrics_keys:
        combined = max(rl_m[key], pd_m[key], 1e-9)
        vals_rl.append(rl_m[key] / combined)
        vals_pd.append(pd_m[key] / combined)
    vals_rl += vals_rl[:1]; vals_pd += vals_pd[:1]

    ax.set_facecolor(PANEL_BG)
    ax.plot(angles, vals_rl, color=RL_COLOR, lw=2)
    ax.fill(angles, vals_rl, color=RL_COLOR, alpha=0.25)
    ax.plot(angles, vals_pd, color=PD_COLOR, lw=2)
    ax.fill(angles, vals_pd, color=PD_COLOR, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8, color=TEXT_COLOR)
    ax.set_ylim(0, 1.1)
    ax.set_title("Relative Metrics Radar\n(lower = better)", pad=20)
    ax.legend(handles=[_label_patch(RL_COLOR, "RL"), _label_patch(PD_COLOR, "PD")],
              loc="upper right", bbox_to_anchor=(1.35, 1.1))


def plot_bar_metrics(ax, rl_m, pd_m, keys):
    """Grouped bar chart of selected metrics."""
    x      = np.arange(len(keys))
    w      = 0.35
    rl_v   = [rl_m[k] for k in keys]
    pd_v   = [pd_m[k] for k in keys]
    b1 = ax.bar(x - w/2, rl_v, w, color=RL_COLOR, alpha=0.85, label="RL")
    b2 = ax.bar(x + w/2, pd_v, w, color=PD_COLOR, alpha=0.85, label="PD")
    ax.set_xticks(x)
    ax.set_xticklabels([k.split("(")[0].strip() for k in keys], rotation=35, ha="right", fontsize=7)
    ax.set_title("Key Metrics Comparison (Bar Chart)")
    ax.set_ylabel("Value")
    ax.legend()
    # value labels on bars
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=6, color=TEXT_COLOR)


def plot_velocity_components(ax_x, ax_y, rl_h, pd_h, dt):
    t_rl = np.arange(len(rl_h["d_vel"])) * dt
    for ax, ch, lbl in [(ax_x, 0, "Vx"), (ax_y, 1, "Vy")]:
        ax.plot(t_rl, gaussian_filter1d(rl_h["t_vel"][:, ch], 3), color=TGT_COLOR, lw=2, ls="--", label="Target")
        ax.plot(t_rl, gaussian_filter1d(rl_h["d_vel"][:, ch], 3), color=RL_COLOR,  lw=2, label="RL")
        ax.plot(t_rl, gaussian_filter1d(pd_h["d_vel"][:, ch], 3), color=PD_COLOR,  lw=2, label="PD")
        ax.set_title(f"{lbl} Component over Time")
        ax.set_xlabel("Time (s)"); ax.set_ylabel(f"{lbl} (m/s)")
        ax.legend()


def plot_reward_breakdown(ax, rl_reward, pd_reward):
    """Single-bar comparison of episode reward."""
    bars = ax.bar(["RL Agent", "PD Baseline"], [rl_reward, pd_reward],
                  color=[RL_COLOR, PD_COLOR], width=0.45, alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.set_title("Episode Total Reward")
    ax.set_ylabel("Cumulative Reward")


def plot_within_threshold(ax, rl_h, pd_h):
    """Stacked bar showing time fractions within distance thresholds."""
    thresholds = [0.25, 0.5, 1.0, 2.0]
    labels     = [f"< {t}m" for t in thresholds]
    rl_e = np.linalg.norm(rl_h["t_pos"][:, :2] - rl_h["d_pos"][:, :2], axis=1)
    pd_e = np.linalg.norm(pd_h["t_pos"][:, :2] - pd_h["d_pos"][:, :2], axis=1)
    rl_f = [np.mean(rl_e < t) * 100 for t in thresholds]
    pd_f = [np.mean(pd_e < t) * 100 for t in thresholds]
    x = np.arange(len(thresholds))
    w = 0.35
    ax.bar(x - w/2, rl_f, w, color=RL_COLOR, alpha=0.85, label="RL")
    ax.bar(x + w/2, pd_f, w, color=PD_COLOR, alpha=0.85, label="PD")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Time Spent Within Distance Thresholds")
    ax.set_ylabel("% of Episode"); ax.set_ylim(0, 110)
    ax.legend()


# ==========================================
# 5. MASTER PLOT GENERATOR
# ==========================================
def plot_comparisons(rl_data: dict, pd_data: dict, traj_name: str,
                     save_dir: str, dt: float,
                     rl_reward: float, pd_reward: float,
                     rl_metrics: dict, pd_metrics: dict):
    """Generate all plot pages for one trajectory."""
    os.makedirs(save_dir, exist_ok=True)

    # ── PAGE 1 : Overview (2×4 grid) ────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(f"Overview  ·  {traj_name.upper()}  ·  RL vs PD",
                 fontsize=18, fontweight="bold", color=ACCENT)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    plot_2d_trajectory(fig.add_subplot(gs[0, 0]),       rl_data, pd_data, traj_name)
    plot_error_time(   fig.add_subplot(gs[0, 1]),       rl_data, pd_data, dt)
    plot_speed_profile(fig.add_subplot(gs[0, 2]),       rl_data, pd_data, dt)
    plot_control_effort(fig.add_subplot(gs[0, 3]),      rl_data, pd_data, dt)
    plot_error_distribution(fig.add_subplot(gs[1, 0]),  rl_data, pd_data)
    plot_rolling_rmse(fig.add_subplot(gs[1, 1]),        rl_data, pd_data, dt)
    plot_cumulative_error(fig.add_subplot(gs[1, 2]),    rl_data, pd_data, dt)
    plot_within_threshold(fig.add_subplot(gs[1, 3]),    rl_data, pd_data)

    plt.savefig(os.path.join(save_dir, f"{traj_name}_page1_overview.png"), bbox_inches="tight")
    plt.close()

    # ── PAGE 2 : Kinematics Detail ───────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(f"Kinematics Detail  ·  {traj_name.upper()}",
                 fontsize=18, fontweight="bold", color=ACCENT)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    plot_per_axis_error(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), rl_data, pd_data, dt)
    plot_velocity_components(fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]), rl_data, pd_data, dt)
    
    # Span the heading error across two columns to fill the gap left by removing jerk
    ax_heading = fig.add_subplot(gs[1, 0:2])
    plot_heading_error(ax_heading, rl_data, pd_data, dt)
    
    plot_action_channels(fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3]), rl_data, pd_data, dt)

    plt.savefig(os.path.join(save_dir, f"{traj_name}_page2_kinematics.png"), bbox_inches="tight")
    plt.close()

    # ── PAGE 3 : Control & Phase Space ──────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(f"Control & Phase Space  ·  {traj_name.upper()}",
                 fontsize=18, fontweight="bold", color=ACCENT)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    ax_ps_rl = fig.add_subplot(gs[0, 0])
    ax_ps_pd = fig.add_subplot(gs[0, 1])
    plot_phase_space(ax_ps_rl, rl_data, RL_COLOR, RL_CMAP, "RL Agent")
    plot_phase_space(ax_ps_pd, pd_data, PD_COLOR, PD_CMAP, "PD Baseline")

    plot_action_scatter(fig.add_subplot(gs[0, 2]),      rl_data, pd_data)
    plot_polar_action(fig.add_subplot(gs[0, 3], polar=True), rl_data, pd_data)

    ax_heat_rl = fig.add_subplot(gs[1, 0])
    ax_heat_pd = fig.add_subplot(gs[1, 1])
    plot_error_heatmap(ax_heat_rl, rl_data, RL_COLOR, "RL Position Error KDE")
    plot_error_heatmap(ax_heat_pd, pd_data, PD_COLOR, "PD Position Error KDE")

    plot_reward_breakdown(fig.add_subplot(gs[1, 2]), rl_reward, pd_reward)
    ax_radar = fig.add_subplot(gs[1, 3], polar=True)
    radar_keys = ["RMSE (m)", "Mean Accel (m/s²)", "Control Effort", "Action Jitter",
                  "Speed Dev (m/s)", "Heading Err (deg)"]
    plot_metrics_radar(ax_radar, rl_metrics, pd_metrics, radar_keys)

    plt.savefig(os.path.join(save_dir, f"{traj_name}_page3_control_phase.png"), bbox_inches="tight")
    plt.close()

    # ── PAGE 4 : 3-D trajectory + bar chart ─────────────────────────────────
    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(f"3-D Trajectory & Metric Summary  ·  {traj_name.upper()}",
                 fontsize=18, fontweight="bold", color=ACCENT)
    ax3d  = fig.add_subplot(121, projection="3d")
    ax3d.set_facecolor(PANEL_BG)
    plot_3d_trajectory(ax3d, rl_data, pd_data)

    ax_bar = fig.add_subplot(122)
    bar_keys = ["RMSE (m)", "Max Error (m)", "Mean Error (m)", "P95 Error (m)",
                "Speed Dev (m/s)", "Mean Accel (m/s²)", "Control Effort", "Action Jitter"]
    plot_bar_metrics(ax_bar, rl_metrics, pd_metrics, bar_keys)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{traj_name}_page4_3d_summary.png"), bbox_inches="tight")
    plt.close()

    logging.info(f"    → 4 plot pages saved to {save_dir}/")


# ==========================================
# 6. CROSS-TRAJECTORY SUMMARY PLOTS
# ==========================================
def plot_cross_trajectory_summary(all_metrics: List[dict], save_dir: str):
    """Generate aggregate plots across all tested trajectories."""
    df = pd.DataFrame(all_metrics)
    trajectories = df["Trajectory"].unique().tolist()
    summary_keys = ["RMSE (m)", "Speed Dev (m/s)", "Control Effort",
                    "Time within 0.5m (%)", "Path Efficiency (%)", "Heading Err (deg)"]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle("Cross-Trajectory Performance Summary  ·  RL vs PD",
                 fontsize=18, fontweight="bold", color=ACCENT)

    for ax, key in zip(axes.flat, summary_keys):
        rl_v = df[df["Agent"] == "RL"][key].values
        pd_v = df[df["Agent"] == "PD"][key].values
        x    = np.arange(len(trajectories))
        w    = 0.35
        ax.bar(x - w/2, rl_v, w, color=RL_COLOR, alpha=0.85, label="RL")
        ax.bar(x + w/2, pd_v, w, color=PD_COLOR, alpha=0.85, label="PD")
        ax.set_xticks(x); ax.set_xticklabels(trajectories, rotation=30, ha="right", fontsize=8)
        ax.set_title(key); ax.set_ylabel(key)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cross_trajectory_summary.png"), bbox_inches="tight")
    plt.close()
    logging.info(f"  → Cross-trajectory summary saved.")


def plot_cross_trajectory_heatmap(all_metrics: List[dict], save_dir: str):
    """Normalised performance heatmap – trajectories × metrics."""
    df = pd.DataFrame(all_metrics)
    pivot_keys = ["RMSE (m)", "Max Error (m)", "Mean Accel (m/s²)",
                  "Control Effort", "Action Jitter", "Speed Dev (m/s)",
                  "Time within 0.5m (%)", "Path Efficiency (%)"]

    for agent_label, color in [("RL", RL_COLOR), ("PD", PD_COLOR)]:
        sub = df[df["Agent"] == agent_label]
        mat = sub[pivot_keys].values.astype(float)
        # Column-wise normalise to [0,1] for consistent colouring
        col_min = mat.min(axis=0, keepdims=True)
        col_max = mat.max(axis=0, keepdims=True)
        with np.errstate(invalid="ignore"):
            mat_norm = (mat - col_min) / np.where(col_max - col_min > 0, col_max - col_min, 1)

        fig, ax = plt.subplots(figsize=(14, max(4, len(sub) * 0.9 + 1)))
        fig.patch.set_facecolor(DARK_BG)
        im = ax.imshow(mat_norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Normalised Value (0=best, 1=worst)")
        ax.set_xticks(range(len(pivot_keys)))
        ax.set_xticklabels([k.split("(")[0].strip() for k in pivot_keys],
                           rotation=40, ha="right", color=TEXT_COLOR, fontsize=9)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["Trajectory"].tolist(), color=TEXT_COLOR)
        for r in range(mat_norm.shape[0]):
            for c in range(mat_norm.shape[1]):
                ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center",
                        fontsize=7, color="black",
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
        ax.set_title(f"{agent_label} Agent – Metric Heatmap (all trajectories)",
                     fontsize=14, fontweight="bold", color=ACCENT)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"heatmap_{agent_label}.png"), bbox_inches="tight")
        plt.close()
        logging.info(f"  → {agent_label} heatmap saved.")


# ==========================================
# 7. MAIN EVALUATION LOOP
# ==========================================
def run_episode(env: DroneTrackingEnv, agent_or_controller, trajectory: str, render: bool) -> Tuple[float, dict, bool]:
    obs, _ = env.reset(seed=42, options={"trajectory_mode": trajectory})
    ep_reward = 0.0
    d_pos, t_pos, d_vel, t_vel, acts = [], [], [], [], []
    if render:
        for _ in range(10): p.stepSimulation()
    done, truncated = False, False
    while not (done or truncated):
        action = agent_or_controller.select_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        d_pos.append(env.drone_pos.copy()); t_pos.append(env.target_pos.copy())
        d_vel.append(env.drone_vel.copy()); t_vel.append(env.target_vel.copy())
        acts.append(action.copy())
        ep_reward += reward; obs = next_obs
        if render:
            p.addUserDebugLine(env.target_pos, env.drone_pos, [0, 1, 0],
                               lineWidth=2.5, lifeTime=env.cfg.control_period)
            time.sleep(env.cfg.control_period)
    history = {
        "d_pos": np.array(d_pos), "t_pos": np.array(t_pos),
        "d_vel": np.array(d_vel), "t_vel": np.array(t_vel), "act": np.array(acts)
    }
    return ep_reward, history, done


def evaluate_checkpoint(checkpoint_path: str, trajectories: List[str], render: bool = True):
    ckpt_dir  = os.path.dirname(checkpoint_path) or "."
    plots_dir = os.path.join(ckpt_dir, "eval_plots")
    os.makedirs(plots_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(ckpt_dir, "advanced_eval.log"), mode="w"),
            logging.StreamHandler()
        ]
    )

    logging.info("=" * 80)
    logging.info(f"  ADVANCED EVALUATION  ·  {checkpoint_path}")
    logging.info("=" * 80)

    env = DroneTrackingEnv(cfg=None, trajectory_mode="square", GUI_mode=render)
    dt  = env.cfg.control_period

    # Init RL Agent
    td3_cfg = TD3Config(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=1.0,
        hidden_dim=256
    )
    rl_agent = TD3Agent(td3_cfg)
    try:
        rl_agent.load(checkpoint_path, strict=False)
        rl_agent.actor.eval()
        logging.info("✔  RL weights loaded.\n")
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        env.close(); return

    pd_controller = BaselinePDController(kp=1.5, kd=0.3)
    all_metrics   = []

    try:
        for traj in trajectories:
            if traj == "teleop":
                logging.warning("Skipping 'teleop' for automated evaluation."); continue
            logging.info(f"{'─'*60}")
            logging.info(f"  Trajectory : {traj.upper()}")
            logging.info(f"{'─'*60}")

            rl_reward, rl_hist, rl_lost = run_episode(env, rl_agent,      traj, render)
            pd_reward, pd_hist, pd_lost = run_episode(env, pd_controller,  traj, render)

            rl_m = compute_advanced_metrics(
                drone_pos=rl_hist["d_pos"], target_pos=rl_hist["t_pos"],
                drone_vel=rl_hist["d_vel"], target_vel=rl_hist["t_vel"],
                actions=rl_hist["act"], dt=dt)
            pd_m = compute_advanced_metrics(
                drone_pos=pd_hist["d_pos"], target_pos=pd_hist["t_pos"],
                drone_vel=pd_hist["d_vel"], target_vel=pd_hist["t_vel"],
                actions=pd_hist["act"], dt=dt)

            logging.info(f"  RL → Reward:{rl_reward:8.1f}  RMSE:{rl_m['RMSE (m)']:5.3f}m  "
                         f"CtrlEff:{rl_m['Control Effort']:5.2f}  PathEff:{rl_m['Path Efficiency (%)']:5.1f}%  Lost:{rl_lost}")
            logging.info(f"  PD → Reward:{pd_reward:8.1f}  RMSE:{pd_m['RMSE (m)']:5.3f}m  "
                         f"CtrlEff:{pd_m['Control Effort']:5.2f}  PathEff:{pd_m['Path Efficiency (%)']:5.1f}%  Lost:{pd_lost}\n")

            for agent_label, metrics, reward, lost in [
                ("RL", rl_m, rl_reward, rl_lost),
                ("PD", pd_m, pd_reward, pd_lost)
            ]:
                row = {"Trajectory": traj, "Agent": agent_label,
                       "Reward": reward, "Target Lost": int(lost)}
                row.update(metrics)
                all_metrics.append(row)

            # Per-trajectory plots (4 pages)
            traj_plot_dir = os.path.join(plots_dir, traj)
            plot_comparisons(rl_hist, pd_hist, traj, traj_plot_dir, dt,
                             rl_reward, pd_reward, rl_m, pd_m)

    except KeyboardInterrupt:
        logging.info("\nEvaluation interrupted. Saving collected data...")
    finally:
        env.close()

    # Save CSV
    df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(ckpt_dir, "advanced_metrics.csv")
    df.to_csv(csv_path, index=False)

    # Cross-trajectory summary plots (only if ≥2 trajectories evaluated)
    if len(df["Trajectory"].unique()) >= 2:
        plot_cross_trajectory_summary(all_metrics, plots_dir)
        plot_cross_trajectory_heatmap(all_metrics, plots_dir)

    logging.info("\n" + "=" * 80)
    logging.info(f"  Metrics CSV : {csv_path}")
    logging.info(f"  Plots dir   : {plots_dir}")
    logging.info("=" * 80)


# ==========================================
# 8. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Evaluation against PD baseline")
    parser.add_argument("--checkpoint",  type=str, default="/home/teaching/RL/checkpoints_spline/final.pt")
    parser.add_argument("--GUI",         action="store_true", help="Enable PyBullet GUI")
    parser.add_argument("--trajectory",  type=str, default="all")
    args = parser.parse_args()

    TEST_TRAJECTORIES = [
        "square", "triangular", "sawtooth", "square_wave",
        "spline_easy", "spline_medium", "spline_hard"
    ] if args.trajectory == "all" else [args.trajectory]

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        trajectories=TEST_TRAJECTORIES,
        render=args.GUI
    )