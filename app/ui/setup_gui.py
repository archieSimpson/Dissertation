import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle


def run_setup_gui(
    plot_limit: float,
    default_food_count: int,
    min_food_count: int,
    max_food_count: int,
    default_food_cluster_spread: float,
    min_food_cluster_spread: float,
):
    selected = {
        "current_food_count": default_food_count,
        "current_center": None,
        "current_spread": default_food_cluster_spread,
        "dragging": False,
        "started": False,
        "patches": [],
    }

    fig = plt.figure(figsize=(9, 10))

    ax_main = fig.add_axes([0.08, 0.30, 0.84, 0.62])
    ax_slider = fig.add_axes([0.15, 0.20, 0.70, 0.04])
    ax_add = fig.add_axes([0.12, 0.08, 0.20, 0.07])
    ax_undo = fig.add_axes([0.40, 0.08, 0.20, 0.07])
    ax_start = fig.add_axes([0.68, 0.08, 0.20, 0.07])

    ax_main.set_xlim(-plot_limit, plot_limit)
    ax_main.set_ylim(-plot_limit, plot_limit)
    ax_main.set_aspect("equal")
    ax_main.grid(True)
    ax_main.set_title("Setup: click and drag to create food patches, then press Start")

    slider = Slider(
        ax=ax_slider,
        label="Food count for next patch",
        valmin=min_food_count,
        valmax=max_food_count,
        valinit=default_food_count,
        valstep=1,
    )

    add_button = Button(ax_add, "Add Patch")
    undo_button = Button(ax_undo, "Undo Last")
    start_button = Button(ax_start, "Start")

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
            preview = (
                f"Preview patch: ({cx:.1f}, {cy:.1f}), "
                f"spread {cr:.1f}, food {cc}"
            )

        total_food = sum(p["count"] for p in selected["patches"])
        info_text.set_text(
            f"Saved patches: {len(selected['patches'])}\n"
            f"Total food: {total_food}\n"
            f"{preview}"
        )

    def on_slider_change(val):
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

    slider.on_changed(on_slider_change)
    add_button.on_clicked(on_add)
    undo_button.on_clicked(on_undo)
    start_button.on_clicked(on_start)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    refresh_info()
    plt.show()

    if not selected["started"]:
        raise RuntimeError("Setup window was closed before starting the simulation.")

    return selected["patches"]