import pandas as pd
import matplotlib.pyplot as plt


def main():
    summary = pd.read_csv("shepherding_log.csv")
    sheep = pd.read_csv("sheep_positions_log.csv")

    total_sheep = sheep["sheep_id"].nunique()

    # 1. Spread over time
    plt.figure(figsize=(10, 5))
    plt.plot(summary["time"], summary["spread"])
    plt.xlabel("Time")
    plt.ylabel("Spread")
    plt.title("Flock Spread Over Time")
    plt.tight_layout()
    plt.savefig("spread_over_time.png")
    plt.close()

    # 2. Polarization over time
    plt.figure(figsize=(10, 5))
    plt.plot(summary["time"], summary["polarization"])
    plt.xlabel("Time")
    plt.ylabel("Polarization")
    plt.title("Flock Polarization Over Time")
    plt.tight_layout()
    plt.savefig("polarization_over_time.png")
    plt.close()

    # 3. Convex hull area over time
    plt.figure(figsize=(10, 5))
    plt.plot(summary["time"], summary["convex_hull_area"])
    plt.xlabel("Time")
    plt.ylabel("Convex Hull Area")
    plt.title("Convex Hull Area Over Time")
    plt.tight_layout()
    plt.savefig("convex_hull_area_over_time.png")
    plt.close()

    # 4. Clusters over time
    plt.figure(figsize=(10, 5))
    plt.plot(summary["time"], summary["num_clusters"])
    plt.xlabel("Time")
    plt.ylabel("Number of Clusters")
    plt.title("Flock Fragmentation Over Time")
    plt.tight_layout()
    plt.savefig("clusters_over_time.png")
    plt.close()

    # 5. Goal progress over time
    plt.figure(figsize=(10, 5))
    plt.plot(summary["time"], summary["in_goal"] / total_sheep)
    plt.xlabel("Time")
    plt.ylabel("Fraction of Sheep in Goal")
    plt.title("Goal Progress Over Time")
    plt.tight_layout()
    plt.savefig("goal_progress_over_time.png")
    plt.close()

    # 6. Dog speed over time
    plt.figure(figsize=(10, 5))
    plt.plot(summary["time"], summary["dog_speed"])
    plt.xlabel("Time")
    plt.ylabel("Dog Speed")
    plt.title("Dog Speed Over Time")
    plt.tight_layout()
    plt.savefig("dog_speed_over_time.png")
    plt.close()

    # 7. Final sheep positions
    final_time = sheep["time"].max()
    final_positions = sheep[sheep["time"] == final_time]

    plt.figure(figsize=(8, 8))
    plt.scatter(final_positions["x"], final_positions["y"], s=20)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Final Sheep Positions")
    plt.tight_layout()
    plt.savefig("final_sheep_positions.png")
    plt.close()

    # 8. Per-sheep trajectories
    plt.figure(figsize=(10, 8))
    for sheep_id, group in sheep.groupby("sheep_id"):
        plt.plot(group["x"], group["y"], linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Per-Sheep Trajectories")
    plt.tight_layout()
    plt.savefig("sheep_trajectories.png")
    plt.close()

    # 9. Alert count over time
    plt.figure(figsize=(10, 5))
    plt.plot(summary["time"], summary["alert_count"])
    plt.xlabel("Time")
    plt.ylabel("Alert Count")
    plt.title("Number of Alert Sheep Over Time")
    plt.tight_layout()
    plt.savefig("alert_count_over_time.png")
    plt.close()

    # 10. Elongation ratio over time
    plt.figure(figsize=(10, 5))
    plt.plot(summary["time"], summary["elongation_ratio"])
    plt.xlabel("Time")
    plt.ylabel("Elongation Ratio")
    plt.title("Flock Elongation Over Time")
    plt.tight_layout()
    plt.savefig("elongation_over_time.png")
    plt.close()

    # 11. Mean nearest-neighbour distance over time
    plt.figure(figsize=(10, 5))
    plt.plot(summary["time"], summary["mean_nn_distance"])
    plt.xlabel("Time")
    plt.ylabel("Mean Nearest-Neighbour Distance")
    plt.title("Neighbour Distance Over Time")
    plt.tight_layout()
    plt.savefig("mean_nn_distance_over_time.png")
    plt.close()

    # 12. Sheep trait distributions
    first_traits = sheep.groupby("sheep_id").first()

    plt.figure(figsize=(8, 5))
    plt.hist(first_traits["fear_strength"], bins=10)
    plt.xlabel("Fear Strength")
    plt.ylabel("Count")
    plt.title("Distribution of Fear Strength")
    plt.tight_layout()
    plt.savefig("fear_strength_hist.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(first_traits["graze_bias"], bins=10)
    plt.xlabel("Graze Bias")
    plt.ylabel("Count")
    plt.title("Distribution of Graze Bias")
    plt.tight_layout()
    plt.savefig("graze_bias_hist.png")
    plt.close()

    # 13. Average sheep speed by sheep
    avg_speed = sheep.groupby("sheep_id")["speed"].mean()

    plt.figure(figsize=(10, 5))
    plt.bar(avg_speed.index.astype(str), avg_speed.values)
    plt.xlabel("Sheep ID")
    plt.ylabel("Average Speed")
    plt.title("Average Speed Per Sheep")
    plt.tight_layout()
    plt.savefig("avg_speed_per_sheep.png")
    plt.close()

    # 14. Fraction of time alert per sheep
    alert_fraction = sheep.groupby("sheep_id")["alert"].mean()

    plt.figure(figsize=(10, 5))
    plt.bar(alert_fraction.index.astype(str), alert_fraction.values)
    plt.xlabel("Sheep ID")
    plt.ylabel("Fraction of Time Alert")
    plt.title("Alert Fraction Per Sheep")
    plt.tight_layout()
    plt.savefig("alert_fraction_per_sheep.png")
    plt.close()

    print("Saved plots:")
    print("spread_over_time.png")
    print("polarization_over_time.png")
    print("convex_hull_area_over_time.png")
    print("clusters_over_time.png")
    print("goal_progress_over_time.png")
    print("dog_speed_over_time.png")
    print("final_sheep_positions.png")
    print("sheep_trajectories.png")
    print("alert_count_over_time.png")
    print("elongation_over_time.png")
    print("mean_nn_distance_over_time.png")
    print("fear_strength_hist.png")
    print("graze_bias_hist.png")
    print("avg_speed_per_sheep.png")
    print("alert_fraction_per_sheep.png")


if __name__ == "__main__":
    main()