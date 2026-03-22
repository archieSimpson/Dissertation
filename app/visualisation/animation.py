import matplotlib.pyplot as plt


def animate_live(simulation, total_steps: int, limit: float, interval: float = 0.03):
    plt.ion()

    fig, ax = plt.subplots(figsize=(10, 10))

    sheep_scat = ax.scatter([], [], s=10, alpha=0.8)
    food_scat = ax.scatter([], [], s=18, alpha=0.9)
    just_ate_scat = ax.scatter([], [], s=42, facecolors="none", linewidths=1.3)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Sheep Foraging Simulation")

    for step in range(total_steps):
        positions, food_positions, food_alive = simulation.step()

        sheep_scat.set_offsets(positions)
        sheep_scat.set_color("tab:blue")

        alive_food = food_positions[food_alive]
        if alive_food.size > 0:
            food_scat.set_offsets(alive_food)
            food_scat.set_color("black")
        else:
            food_scat.set_offsets([[-10_000, -10_000]])

        just_ate_pos = positions[simulation.just_ate]
        if just_ate_pos.size > 0:
            just_ate_scat.set_offsets(just_ate_pos)
            just_ate_scat.set_edgecolor("red")
        else:
            just_ate_scat.set_offsets([[-10_000, -10_000]])

        ax.set_title(f"Sheep Foraging Simulation - step {step + 1}/{total_steps}")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(interval)

    plt.ioff()
    plt.close(fig)