import numpy as np
from scipy.spatial import ConvexHull, QhullError

from flock import centroid, count_clusters, mean_nearest_neighbour_distance, polarization, spread


def convex_hull_area(sheep_list) -> float:
    if len(sheep_list) < 3:
        return 0.0
    points = np.array([s.pos for s in sheep_list])
    try:
        hull = ConvexHull(points)
        return float(hull.volume)  # in 2D scipy uses volume for polygon area
    except QhullError:
        return 0.0


def elongation_ratio(sheep_list) -> float:
    if len(sheep_list) < 3:
        return 1.0

    points = np.array([s.pos for s in sheep_list])
    centered = points - np.mean(points, axis=0)
    cov = np.cov(centered.T)

    vals, _ = np.linalg.eigh(cov)
    vals = np.sort(vals)

    if vals[0] <= 1e-8:
        return float("inf")
    return float(vals[1] / vals[0])


def compute_metrics(sheep_list, dog, env, cfg, sim_time: float):
    c = centroid(sheep_list)
    s = spread(sheep_list)
    p = polarization(sheep_list)
    mnn = mean_nearest_neighbour_distance(sheep_list)
    in_goal = env.sheep_in_goal(sheep_list)
    alert_count = sum(1 for sh in sheep_list if sh.alert)

    hull_area = convex_hull_area(sheep_list)
    elongation = elongation_ratio(sheep_list)
    num_clusters = count_clusters(sheep_list, cfg.cluster_distance_threshold)

    avg_fear = float(np.mean([sh.fear_strength for sh in sheep_list]))
    avg_graze_bias = float(np.mean([sh.graze_bias for sh in sheep_list]))
    avg_reaction_delay = float(np.mean([sh.reaction_delay_frames for sh in sheep_list]))
    avg_sheep_speed = float(np.mean([np.linalg.norm(sh.vel) for sh in sheep_list]))

    return {
        "time": sim_time,
        "centroid_x": float(c[0]),
        "centroid_y": float(c[1]),
        "spread": float(s),
        "polarization": float(p),
        "mean_nn_distance": float(mnn),
        "convex_hull_area": float(hull_area),
        "elongation_ratio": float(elongation),
        "num_clusters": int(num_clusters),
        "dog_x": float(dog.pos[0]),
        "dog_y": float(dog.pos[1]),
        "dog_vx": float(dog.vel[0]),
        "dog_vy": float(dog.vel[1]),
        "dog_speed": float(np.linalg.norm(dog.vel)),
        "dog_mode": dog.mode,
        "dog_submode": dog.submode,
        "in_goal": int(in_goal),
        "alert_count": int(alert_count),
        "avg_sheep_speed": float(avg_sheep_speed),
        "avg_fear_strength": float(avg_fear),
        "avg_graze_bias": float(avg_graze_bias),
        "avg_reaction_delay": float(avg_reaction_delay),
        "success": int(in_goal == len(sheep_list)),
    }