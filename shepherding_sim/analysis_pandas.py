import pandas as pd
import matplotlib.pyplot as plt


def main():
    summary = pd.read_csv("shepherding_log.csv")
    sheep = pd.read_csv("sheep_positions_log.csv")

    print("\n=== Summary columns ===")
    print(summary.columns.tolist())

    print("\n=== Final summary row ===")
    print(summary.iloc[-1])

    print("\n=== Basic stats ===")
    print(summary[[
        "spread",
        "polarization",
        "convex_hull_area",
        "elongation_ratio",
        "num_clusters",
        "dog_speed",
        "alert_count",
        "avg_sheep_speed",
    ]].describe())

    print("\n=== Per-sheep trait snapshot ===")
    sheep_traits = sheep.groupby("sheep_id").first()[[
        "fear_strength", "max_speed_multiplier", "reaction_delay_frames", "graze_bias"
    ]]
    print(sheep_traits.describe())

    # Plot 1: spread over time
    plt.figure(figsize=(10, 5))
    plt.plot(summary["time"], summary["spread"])
    plt.xlabel("Time")
    plt.ylabel("Spread")
    plt.title("Flock Spread Over Time")
    plt.tight_layout()
    plt.savefig("plot_spread.png")
    plt.close()

    # Plot 2: polarization over time
    plt.figure(figsize=(10, 5))
    plt.plot(summary["time"], summary["polarization"])
    plt.xlabel("Time")
    plt.ylabel("Polarization")
    plt.title("Flock Polarization Over Time")
    plt.tight_layout()
    plt.savefig("plot_polarization.png")
    plt.close()

    # Plot 3: hull area over time
    plt.figure(figsize=(10, 5))
    plt.plot(summary["time"], summary["convex_hull_area"])
    plt.xlabel("Time")
    plt.ylabel("Convex Hull Area")
    plt.title("Flock Hull Area Over Time")
    plt.tight_layout()
    plt.savefig("plot_hull_area.png")
    plt.close()

    # Plot 4: number of clusters
    plt.figure(figsize=(10, 5))
    plt.plot(summary["time"], summary["num_clusters"])
    plt.xlabel("Time")
    plt.ylabel("Clusters")
    plt.title("Number of Flock Clusters Over Time")
    plt.tight_layout()
    plt.savefig("plot_clusters.png")
    plt.close()

    # Final sheep positions
    final_time = sheep["time"].max()
    final_positions = sheep[sheep["time"] == final_time]

    plt.figure(figsize=(8, 8))
    plt.scatter(final_positions["x"], final_positions["y"], s=20)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Final Sheep Positions")
    plt.tight_layout()
    plt.savefig("plot_final_positions.png")
    plt.close()

    print("\nSaved:")
    print("  plot_spread.png")
    print("  plot_polarization.png")
    print("  plot_hull_area.png")
    print("  plot_clusters.png")
    print("  plot_final_positions.png")


if __name__ == "__main__":
    main()