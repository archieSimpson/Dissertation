"""§4.3 Data Collection — comprehensive metrics extraction for all four
subsections (uniform_high, uniform_low, radial_increase, corridors).

Resolves positions.csv/metrics.csv from either outputs/<scenario>_seed{N}/
or outputs/final_frames/<scenario>/ (whichever exists). Ablation runs and
multi-seed runs are expected at fixed paths populated by
run_section_4_3_sims.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUTS = Path("outputs")
N_SHEEP = 40
W, H = 220.0, 140.0
FIELD_CX, FIELD_CY = W / 2, H / 2


def find_run_dir(scenario: str, seed: int, suffix: str = "") -> Path:
    """Locate the output directory containing positions.csv for this run."""
    if suffix:
        return OUTPUTS / f"{scenario}_seed{seed}{suffix}"
    candidates = [
        OUTPUTS / f"{scenario}_seed{seed}",
        OUTPUTS / "final_frames" / scenario,
    ]
    for d in candidates:
        if (d / "positions.csv").exists():
            return d
    raise FileNotFoundError(f"No positions.csv for {scenario} seed {seed}")


def load(scenario: str, seed: int, suffix: str = "") -> dict:
    d = find_run_dir(scenario, seed, suffix)
    out = {"dir": d, "positions": pd.read_csv(d / "positions.csv")}
    if (d / "metrics.csv").exists():
        out["metrics"] = pd.read_csv(d / "metrics.csv")
    if (d / "field_health.csv").exists():
        out["field"] = pd.read_csv(d / "field_health.csv")
    return out


def state_percentages(pos: pd.DataFrame) -> dict[str, float]:
    """% of agent-steps in each state."""
    total = len(pos)
    return {st: float(100.0 * (pos["state"] == st).sum() / total)
            for st in ("grazing", "walking", "travelling", "regrouping", "resting")}


def cluster_count_mean(metrics: pd.DataFrame) -> float | None:
    if "number_of_clusters" not in metrics.columns:
        return None
    return float(metrics["number_of_clusters"].mean())


def spawn_centre_bias(pos: pd.DataFrame, near: float = 30.0, far: float = 60.0) -> dict[str, float]:
    dx = pos["x"] - FIELD_CX
    dy = pos["y"] - FIELD_CY
    dist = np.sqrt(dx ** 2 + dy ** 2)
    n_near = float((dist < near).mean() * 100)
    n_far  = float((dist > far).mean()  * 100)
    return {"near_centre_pct": n_near, "near_margin_pct": n_far,
            "ratio": n_near / max(n_far, 1e-9)}


def daily_distance(pos: pd.DataFrame) -> float:
    last_path = pos.groupby("sheep_id")["path_length"].max()
    return float(last_path.mean())


def section_header(title: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")



section_header("§4.3.1  Uniform High  (abundant params on uniform NDVI = 1.0)")
data = load("uniform_high", 42)
pos = data["positions"]
metrics = data.get("metrics")

print("State percentages (agent-step weighted):")
for st, pct in state_percentages(pos).items():
    print(f"   {st:11s} {pct:5.1f}%")

if metrics is not None:
    print(f"\nMean cluster count : {cluster_count_mean(metrics):.2f}")

bias = spawn_centre_bias(pos)
print(f"\nTime near centre (<30m) : {bias['near_centre_pct']:5.1f}%")
print(f"Time near margin (>60m) : {bias['near_margin_pct']:5.1f}%")
print(f"Centre / margin ratio   : {bias['ratio']:.2f}")



section_header("§4.3.2  Uniform Low  (scarce params on uniform NDVI = 0.1)")
data    = load("uniform_low", 42)
data_np = load("uniform_low", 42, suffix="_nopersonality")
pos    = data["positions"]
pos_np = data_np["positions"]

print("State percentages (full features):")
sp_full = state_percentages(pos)
for st, pct in sp_full.items():
    print(f"   {st:11s} {pct:5.1f}%")

print("\nState percentages (no personality):")
sp_no = state_percentages(pos_np)
for st, pct in sp_no.items():
    print(f"   {st:11s} {pct:5.1f}%")

print(f"\nMean daily distance (full)         : {daily_distance(pos):.0f} m")
print(f"Mean daily distance (no personality): {daily_distance(pos_np):.0f} m")

if data.get("metrics") is not None:
    print(f"\nMean cluster count (full)         : {cluster_count_mean(data['metrics']):.2f}")
if data_np.get("metrics") is not None:
    print(f"Mean cluster count (no personality): {cluster_count_mean(data_np['metrics']):.2f}")

print(f"\nREGROUPING delta : full {sp_full['regrouping']:.1f}%  ->  "
      f"no personality {sp_no['regrouping']:.1f}%  "
      f"(delta {sp_no['regrouping'] - sp_full['regrouping']:+.1f}%)")



section_header("§4.3.3  Radial Increase  (abundant params, rich edges)")
data    = load("radial_increase", 42)
data_ns = load("radial_increase", 42, suffix="_nosocial")
pos    = data["positions"]
pos_ns = data_ns["positions"]

print("State percentages (full features):")
for st, pct in state_percentages(pos).items():
    print(f"   {st:11s} {pct:5.1f}%")

def corner_counts(positions: pd.DataFrame, radius: float = 30.0) -> dict[str, int]:
    last = positions[positions["step"] == positions["step"].max()]
    corners = {"BL": (0, 0), "BR": (W, 0), "TL": (0, H), "TR": (W, H)}
    out = {}
    for name, (cx, cy) in corners.items():
        d = np.sqrt((last["x"] - cx) ** 2 + (last["y"] - cy) ** 2)
        out[name] = int((d < radius).sum())
    out["interior"] = N_SHEEP - sum(out.values())
    return out

print("\nFinal corner partition (full features):")
for name, n in corner_counts(pos).items():
    print(f"   {name:8s} {n:2d}")

print("\nFinal corner partition (no social):")
for name, n in corner_counts(pos_ns).items():
    print(f"   {name:8s} {n:2d}")

if data.get("field") is not None:
    last = data["field"].iloc[-1]
    print(f"\nFinal field health (full)  : {last['field_health_pct']:.1f}%")
    print(f"Final degraded area (full) : {last['degraded_area_pct']:.1f}%")
if data_ns.get("field") is not None:
    last = data_ns["field"].iloc[-1]
    print(f"Final field health (no soc): {last['field_health_pct']:.1f}%")
    print(f"Final degraded area (no soc): {last['degraded_area_pct']:.1f}%")



section_header("§4.3.4  Corridors  (cascade scenario)")
data    = load("corridors", 42)
data_ns = load("corridors", 42, suffix="_nosocial")
pos    = data["positions"]
pos_ns = data_ns["positions"]
metrics    = data.get("metrics")
metrics_ns = data_ns.get("metrics")

def cascade_start(m: pd.DataFrame, threshold: int = 5) -> int | None:
    if "fresh_walkers" not in m.columns:
        return None
    hits = m[m["fresh_walkers"] > threshold]
    return int(hits["step"].iloc[0]) if len(hits) else None

def cascade_peak(m: pd.DataFrame) -> int | None:
    if "fresh_walkers" not in m.columns:
        return None
    return int(m["fresh_walkers"].max())

def time_to_partition(m: pd.DataFrame, threshold: int = 2, sustain: int = 30) -> int | None:
    """First step where number_of_clusters >= threshold and stays for `sustain` steps."""
    if "number_of_clusters" not in m.columns:
        return None
    cc = m["number_of_clusters"].to_numpy()
    for i in range(len(cc) - sustain):
        if (cc[i : i + sustain] >= threshold).all():
            return int(m["step"].iloc[i])
    return None

print("State percentages (full features):")
for st, pct in state_percentages(pos).items():
    print(f"   {st:11s} {pct:5.1f}%")

if metrics is not None:
    print(f"\nTime to cascade start (fresh_walkers > 5) : {cascade_start(metrics)}")
    print(f"Cascade peak intensity (max fresh_walkers): {cascade_peak(metrics)}")
    print(f"Time to partition (clusters >= 2 sustained): {time_to_partition(metrics)}")

if metrics_ns is not None:
    print(f"\nNo-social cascade peak  : {cascade_peak(metrics_ns)}")
    print(f"No-social cascade start : {cascade_start(metrics_ns)}")


print(f"\nMulti-seed final partition (left x<110 vs right x>=110):")
seeds_run = range(42, 51)
splits = []
for seed in seeds_run:
    try:
        d = load("corridors", seed)
    except FileNotFoundError:
        print(f"   seed {seed}: positions missing")
        continue
    p = d["positions"]
    last = p[p["step"] == p["step"].max()]
    L = int((last["x"] < 110).sum())
    R = int((last["x"] >= 110).sum())
    splits.append((seed, L, R))
    print(f"   seed {seed}: {L:2d} left / {R:2d} right   |L-R|={abs(L-R):2d}")

if splits:
    L_arr = np.array([s[1] for s in splits])
    R_arr = np.array([s[2] for s in splits])
    print(f"\n   mean: {L_arr.mean():.1f} left / {R_arr.mean():.1f} right")
    print(f"   SD of left count   : {L_arr.std(ddof=1):.2f}")
    print(f"   most even seed     : {splits[int(np.argmin(np.abs(L_arr - R_arr)))][0]}  "
          f"({L_arr[np.argmin(np.abs(L_arr - R_arr))]} / {R_arr[np.argmin(np.abs(L_arr - R_arr))]})")

print()
