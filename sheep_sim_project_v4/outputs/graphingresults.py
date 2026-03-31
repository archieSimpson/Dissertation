from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RUN_DIR = Path("outputs/abundant")
TIME_PER_STEP_SECONDS = 30


pos = pd.read_csv(RUN_DIR / "positions.csv")
grp = pd.read_csv(RUN_DIR / "metrics.csv")

pos = pos.sort_values(["sheep_id", "step"]).copy()


pos["dx"] = pos.groupby("sheep_id")["x"].diff()
pos["dy"] = pos.groupby("sheep_id")["y"].diff()

pos["step_length"] = np.sqrt(pos["dx"]**2 + pos["dy"]**2)

pos["heading"] = np.arctan2(pos["dy"], pos["dx"])
pos["turn_angle"] = pos.groupby("sheep_id")["heading"].diff()

pos["turn_angle"] = (pos["turn_angle"] + np.pi) % (2 * np.pi) - np.pi
pos["abs_turn_angle"] = pos["turn_angle"].abs()

pos["speed"] = pos["step_length"] / TIME_PER_STEP_SECONDS

pos["hour"] = (pos["step"] * TIME_PER_STEP_SECONDS) / 3600
pos["hour_bin"] = pos["hour"].astype(int)

grp["hour"] = (grp["step"] * TIME_PER_STEP_SECONDS) / 3600


plt.figure()

total_states = (
    grp["grazing_count"]
    + grp["walking_count"]
    + grp["travelling_count"]
    + grp["regrouping_count"]
    + grp["resting_count"]
)

plt.plot(grp["hour"], grp["grazing_count"] / total_states, label="Grazing")
plt.plot(grp["hour"], grp["walking_count"] / total_states, label="Walking")
plt.plot(grp["hour"], grp["travelling_count"] / total_states, label="Travelling")
plt.plot(grp["hour"], grp["regrouping_count"] / total_states, label="Regrouping")
plt.plot(grp["hour"], grp["resting_count"] / total_states, label="Resting")

plt.xlabel("Simulated Hour")
plt.ylabel("Proportion")
plt.title("State Budget Over Time")
plt.legend()
plt.grid()

plt.savefig(RUN_DIR / "state_budget.png", dpi=300)


plt.figure()

speed_by_hour = pos.groupby("hour_bin")["speed"].mean()

plt.plot(speed_by_hour.index, speed_by_hour.values)

plt.xlabel("Hour")
plt.ylabel("Mean Speed")
plt.title("Mean Speed by Hour")
plt.grid()

plt.savefig(RUN_DIR / "mean_speed_by_hour.png", dpi=300)

plt.figure()

plt.hist(pos["step_length"].dropna(), bins=50)

plt.xlabel("Step Length")
plt.ylabel("Frequency")
plt.title("Step Length Distribution")

plt.savefig(RUN_DIR / "step_length_hist.png", dpi=300)


plt.figure()

plt.hist(pos["turn_angle"].dropna(), bins=50)

plt.xlabel("Turning Angle (radians)")
plt.ylabel("Frequency")
plt.title("Turning Angle Distribution")

plt.savefig(RUN_DIR / "turning_angle_hist.png", dpi=300)


plt.figure()

plt.plot(grp["hour"], grp["number_of_clusters"])

plt.xlabel("Simulated Hour")
plt.ylabel("Number of Clusters")
plt.title("Cluster Count Over Time")
plt.grid()

plt.savefig(RUN_DIR / "cluster_count.png", dpi=300)

plt.show()