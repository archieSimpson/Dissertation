import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons
from matplotlib.patches import Circle


def run_setup_gui(
    plot_limit: float,
    default_food_count: int,
    min_food_count: int,
    max_food_count: int,
    default_food_cluster_spread: float,
    min_food_cluster_spread: float,
    default_behaviour_settings: dict,
):
    selected = {
        "current_food_count": default_food_count,
        "current_center": None,
        "current_spread": default_food_cluster_spread,
        "dragging": False,
        "started": False,
        "patches": [],
        "behaviour": default_behaviour_settings.copy(),
    }

    fig = plt.figure(figsize=(14, 10))

    ax_main = fig.add_axes([0.05, 0.28, 0.46, 0.64])
    ax_slider_food = fig.add_axes([0.10, 0.20, 0.34, 0.03])

    ax_add = fig.add_axes([0.05, 0.08, 0.12, 0.06])
    ax_undo = fig.add_axes([0.19, 0.08, 0.12, 0.06])
    ax_start = fig.add_axes([0.33, 0.08, 0.12, 0.06])

    ax_checks = fig.add_axes([0.58, 0.62, 0.18, 0.22])

    ax_rand = fig.add_axes([0.58, 0.54, 0.30, 0.025])
    ax_scent = fig.add_axes([0.58, 0.49, 0.30, 0.025])
    ax_social = fig.add_axes([0.58, 0.44, 0.30, 0.025])
    ax_levy = fig.add_axes([0.58, 0.39, 0.30, 0.025])
    ax_memory = fig.add_axes([0.58, 0.34, 0.30, 0.025])
    ax_forage_pull = fig.add_axes([0.58, 0.29, 0.30, 0.025])
    ax_forage_tangent = fig.add_axes([0.58, 0.24, 0.30, 0.025])
    ax_forage_speed = fig.add_axes([0.58, 0.19, 0.30, 0.025])
    ax_reset_steps = fig.add_axes([0.58, 0.14, 0.30, 0.025])
    ax_min_turn = fig.add_axes([0.58, 0.09, 0.30, 0.025])

    ax_main.set_xlim(-plot_limit, plot_limit)
    ax_main.set_ylim(-plot_limit, plot_limit)
    ax_main.set_aspect("equal")
    ax_main.grid(True)
    ax_main.set_title("Food patch setup: click and drag to create patches")

    slider_food = Slider(
        ax=ax_slider_food,
        label="Food count for next patch",
        valmin=min_food_count,
        valmax=max_food_count,
        valinit=default_food_count,
        valstep=1,
    )

    add_button = Button(ax_add, "Add Patch")
    undo_button = Button(ax_undo, "Undo Last")
    start_button = Button(ax_start, "Start")

    labels = [
        "Scent",
        "Foraging",
        "Social",
        "Levy",
        "Memory",
        "Boundary",
    ]
    actives = [
        selected["behaviour"]["enable_scent"],
        selected["behaviour"]["enable_foraging"],
        selected["behaviour"]["enable_social_signal"],
        selected["behaviour"]["enable_levy_flight"],
        selected["behaviour"]["enable_memory_revisit"],
        selected["behaviour"]["enable_boundary_reflection"],
    ]
    checks = CheckButtons(ax_checks, labels, actives)

    slider_rand = Slider(ax_rand, "Randomness", 0.0, 1.5, valinit=selected["behaviour"]["direction_noise_scale"])
    slider_scent = Slider(ax_scent, "Scent strength", 0.0, 1.0, valinit=selected["behaviour"]["scent_strength"])
    slider_social = Slider(ax_social, "Social strength", 0.0, 1.0, valinit=selected["behaviour"]["social_signal_strength"])
    slider_levy = Slider(ax_levy, "Levy probability", 0.0, 0.25, valinit=selected["behaviour"]["levy_flight_probability"])
    slider_memory = Slider(ax_memory, "Memory revisit", 0.0, 1.0, valinit=selected["behaviour"]["memory_revisit_strength"])
    slider_forage_pull = Slider(ax_forage_pull, "Forage pull", 0.0, 1.0, valinit=selected["behaviour"]["forage_pull_strength"])
    slider_forage_tangent = Slider(ax_forage_tangent, "Forage tangent", 0.0, 1.0, valinit=selected["behaviour"]["forage_tangent_strength"])
    slider_forage_speed = Slider(ax_forage_speed, "Forage speed factor", 0.2, 1.2, valinit=selected["behaviour"]["forage_speed_reduction"])
    slider_reset_steps = Slider(ax_reset_steps, "Reset steps", 10, 200, valinit=selected["behaviour"]["base_direction_reset_steps"], valstep=1)
    slider_min_turn = Slider(ax_min_turn, "Min turn degrees", 0, 180, valinit=selected["behaviour"]["min_direction_change_degrees"], valstep=1)

    preview_marker = ax_main.scatter([], [], s=120, alpha=0.9)
    preview_circle = Circle((0, 0), radius=0.0, fill=False, linewidth=2, alpha=0.8, linestyle="--")
    ax_main.add_patch(preview_circle)
    preview_circle.set_visible(False)

    info_text = ax_main.text(
        0.02,
        0.98,
        "",
        transform=ax_main.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    patch_artists = []

    def redraw_saved_patches():
        nonlocal patch_artists
        for artist in patch_artists:
            artist.remove()
        patch_artists = []

        for i, patch in enumerate(selected["patches"], start=1):
            x, y = patch["center"]
            r = patch["spread"]
            c = Circle((x, y), radius=r, fill=False, linewidth=2, alpha=0.8)
            ax_main.add_patch(c)
            patch_artists.append(c)

            s = ax_main.scatter([x], [y], s=90, alpha=0.9)
            patch_artists.append(s)

            t = ax_main.text(x, y, str(i), ha="center", va="center")
            patch_artists.append(t)

    def refresh_info():
        preview = "No preview patch"
        if selected["current_center"] is not None:
            cx, cy = selected["current_center"]
            cr = selected["current_spread"]
            cc = selected["current_food_count"]
            preview = f"Preview: ({cx:.1f}, {cy:.1f}), spread {cr:.1f}, food {cc}"

        total_food = sum(p["count"] for p in selected["patches"])
        info_text.set_text(
            f"Saved patches: {len(selected['patches'])}\n"
            f"Total food: {total_food}\n"
            f"{preview}"
        )

    def on_slider_food_change(val):
        selected["current_food_count"] = int(val)
        refresh_info()
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax_main or event.xdata is None or event.ydata is None:
            return
        x = float(event.xdata)
        y = float(event.ydata)

        selected["current_center"] = (x, y)
        selected["current_spread"] = default_food_cluster_spread
        selected["dragging"] = True

        preview_marker.set_offsets([[x, y]])
        preview_circle.center = (x, y)
        preview_circle.radius = default_food_cluster_spread
        preview_circle.set_visible(True)

        refresh_info()
        fig.canvas.draw_idle()

    def on_motion(event):
        if not selected["dragging"]:
            return
        if event.inaxes != ax_main or event.xdata is None or event.ydata is None:
            return
        if selected["current_center"] is None:
            return

        cx, cy = selected["current_center"]
        dx = float(event.xdata) - cx
        dy = float(event.ydata) - cy
        radius = max(min_food_cluster_spread, math.hypot(dx, dy))

        selected["current_spread"] = radius
        preview_circle.radius = radius

        refresh_info()
        fig.canvas.draw_idle()

    def on_release(event):
        if selected["dragging"]:
            selected["dragging"] = False
            refresh_info()
            fig.canvas.draw_idle()

    def on_add(event):
        if selected["current_center"] is None:
            return

        selected["patches"].append(
            {
                "center": selected["current_center"],
                "spread": selected["current_spread"],
                "count": selected["current_food_count"],
            }
        )

        redraw_saved_patches()

        selected["current_center"] = None
        selected["current_spread"] = default_food_cluster_spread
        preview_marker.set_offsets([[-10_000, -10_000]])
        preview_circle.set_visible(False)

        refresh_info()
        fig.canvas.draw_idle()

    def on_undo(event):
        if selected["patches"]:
            selected["patches"].pop()
            redraw_saved_patches()
            refresh_info()
            fig.canvas.draw_idle()

    def on_start(event):
        if not selected["patches"]:
            info_text.set_text("Add at least one patch before starting")
            fig.canvas.draw_idle()
            return
        selected["started"] = True
        plt.close(fig)

    def on_check(label):
        mapping = {
            "Scent": "enable_scent",
            "Foraging": "enable_foraging",
            "Social": "enable_social_signal",
            "Levy": "enable_levy_flight",
            "Memory": "enable_memory_revisit",
            "Boundary": "enable_boundary_reflection",
        }
        key = mapping[label]
        selected["behaviour"][key] = not selected["behaviour"][key]

    def bind_slider(slider, key, cast=float):
        def _update(val):
            selected["behaviour"][key] = cast(val)
        slider.on_changed(_update)

    slider_food.on_changed(on_slider_food_change)
    add_button.on_clicked(on_add)
    undo_button.on_clicked(on_undo)
    start_button.on_clicked(on_start)
    checks.on_clicked(on_check)

    bind_slider(slider_rand, "direction_noise_scale", float)
    bind_slider(slider_scent, "scent_strength", float)
    bind_slider(slider_social, "social_signal_strength", float)
    bind_slider(slider_levy, "levy_flight_probability", float)
    bind_slider(slider_memory, "memory_revisit_strength", float)
    bind_slider(slider_forage_pull, "forage_pull_strength", float)
    bind_slider(slider_forage_tangent, "forage_tangent_strength", float)
    bind_slider(slider_forage_speed, "forage_speed_reduction", float)
    bind_slider(slider_reset_steps, "base_direction_reset_steps", int)
    bind_slider(slider_min_turn, "min_direction_change_degrees", float)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    refresh_info()
    plt.show()

    if not selected["started"]:
        raise RuntimeError("Setup window was closed before starting the simulation.")

    return selected["patches"], selected["behaviour"]