from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

def compute_step_lengths(pos_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, grp in pos_df.sort_values("step").groupby("sheep_id"):
        xs = grp["x"].to_numpy()
        ys = grp["y"].to_numpy()
        steps = grp["step"].to_numpy()
        states = grp["state"].to_numpy()
        dists = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
        for i, d in enumerate(dists):
            rows.append({"sheep_id": sid, "step": steps[i + 1], "state": states[i + 1], "step_length": d})
    return pd.DataFrame(rows)

def compute_turning_angles(pos_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, grp in pos_df.sort_values("step").groupby("sheep_id"):
        xs = grp["x"].to_numpy()
        ys = grp["y"].to_numpy()
        steps = grp["step"].to_numpy()
        states = grp["state"].to_numpy()
        dx = np.diff(xs)
        dy = np.diff(ys)
        angles = np.arctan2(dy, dx)
        turning = np.diff(angles)
        turning = (turning + math.pi) % (2 * math.pi) - math.pi
        for i, ta in enumerate(turning):
            rows.append({
                "sheep_id": sid, "step": steps[i + 2],
                "state": states[i + 2], "turning_angle": ta,
            })
    return pd.DataFrame(rows)

def compute_nsd(pos_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, grp in pos_df.sort_values("step").groupby("sheep_id"):
        x0, y0 = grp["x"].iloc[0], grp["y"].iloc[0]
        for _, row in grp.iterrows():
            nsd = (row["x"] - x0) ** 2 + (row["y"] - y0) ** 2
            rows.append({"sheep_id": sid, "step": row["step"], "nsd": nsd})
    return pd.DataFrame(rows)

def circadian_summary(group_df: pd.DataFrame, day_length_steps: int = 1920) -> pd.DataFrame:
    group_df = group_df.copy()
    group_df["day"] = group_df["step"] // day_length_steps
    rows = []
    for day, grp in group_df.groupby("day"):
        t = grp["step"].to_numpy() % day_length_steps
        y = grp["mean_speed"].to_numpy()
        if len(y) < 4:
            continue
        omega = 2 * math.pi / day_length_steps
        A = np.column_stack([np.ones_like(t), np.cos(omega * t), np.sin(omega * t)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        mesor, beta, gamma = coeffs
        amplitude = math.sqrt(beta ** 2 + gamma ** 2)
        acrophase  = (-math.atan2(gamma, beta)) % (2 * math.pi)
        acrophase_step = acrophase / omega
        y_pred = A @ coeffs
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        rows.append({
            "day": int(day), "amplitude": amplitude,
            "acrophase_step": acrophase_step, "mesor": mesor, "r2": r2,
        })
    return pd.DataFrame(rows)

def state_transition_matrix(pos_df: pd.DataFrame) -> pd.DataFrame:
    from_states, to_states = [], []
    for _, grp in pos_df.sort_values("step").groupby("sheep_id"):
        states = grp["state"].to_numpy()
        from_states.extend(states[:-1])
        to_states.extend(states[1:])

    all_states = sorted(set(from_states) | set(to_states))
    matrix = pd.DataFrame(0, index=all_states, columns=all_states)
    for f, t in zip(from_states, to_states):
        matrix.loc[f, t] += 1
    return matrix

def utilisation_grid(
    pos_df: pd.DataFrame,
    field_width: float,
    field_height: float,
    grid_cols: int = 44,
    grid_rows: int = 28,
    bandwidth: float | None = None,
) -> np.ndarray:
    from scipy.stats import gaussian_kde

    xs = pos_df["x"].to_numpy()
    ys = pos_df["y"].to_numpy()

    if bandwidth is None:
        n = len(xs)
        bandwidth = n ** (-1.0 / 6.0)

    kde = gaussian_kde(np.vstack([xs, ys]), bw_method=bandwidth)

    grid_xs = np.linspace(0, field_width, grid_cols)
    grid_ys = np.linspace(0, field_height, grid_rows)
    gx, gy = np.meshgrid(grid_xs, grid_ys)
    coords  = np.vstack([gx.ravel(), gy.ravel()])
    density = kde(coords).reshape(grid_rows, grid_cols)
    return density / density.sum()

def per_individual_summary(pos_df: pd.DataFrame) -> pd.DataFrame:
    last = pos_df.groupby("sheep_id").last().reset_index()
    time_budget = (
        pos_df.groupby(["sheep_id", "state"]).size()
        .unstack(fill_value=0)
        .apply(lambda r: r / r.sum(), axis=1)
        .add_prefix("frac_")
        .reset_index()
    )
    mean_speed = pos_df.groupby("sheep_id")["speed"].mean().reset_index(name="mean_speed")
    result = (
        last[["sheep_id", "path_length", "cumulative_food"]]
        .merge(time_budget, on="sheep_id")
        .merge(mean_speed, on="sheep_id")
    )
    return result.set_index("sheep_id")

def field_degradation_timeline(field_df: pd.DataFrame) -> pd.DataFrame:
    df = field_df.copy().sort_values("step")
    for col in ["field_health_pct", "field_biomass_pct", "degraded_area_pct", "mean_grazing_pressure"]:
        if col in df.columns:
            df[f"{col}_smooth"] = df[col].rolling(window=20, min_periods=1, center=True).mean()
    return df

def cohesion_timeline(group_df: pd.DataFrame) -> pd.DataFrame:
    df = group_df.copy().sort_values("step")
    for col in ["spread", "polarization", "mean_nearest_neighbour_distance", "number_of_clusters"]:
        if col in df.columns:
            df[f"{col}_smooth"] = df[col].rolling(window=20, min_periods=1, center=True).mean()
    return df

def state_budget_timeseries(group_df: pd.DataFrame, n_sheep: int = 40) -> pd.DataFrame:
    df = group_df.copy().sort_values("step")
    for state in ["grazing", "walking", "travelling", "regrouping", "resting"]:
        col = f"{state}_count"
        if col in df.columns:
            df[f"frac_{state}"] = df[col] / n_sheep
    return df
