import numpy as np
import matplotlib.pyplot as plt
from nuscenes.prediction import PredictHelper
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm


def get_yaw(ann):
    q = Quaternion(ann["rotation"])
    return q.yaw_pitch_roll[0]

def get_ann_timestamp(nusc, ann):
    """Return the timestamp (µs) of the sample this annotation belongs to."""
    sample = nusc.get("sample", ann["sample_token"])
    return sample["timestamp"]


def draw_car_box(ax, center, size, yaw, color="red", linewidth=2, zorder=3):
    """Draw a rotated bounding box for a car."""
    width, length = size[0], size[1]
    corners = np.array([
        [ length/2,  width/2],
        [ length/2, -width/2],
        [-length/2, -width/2],
        [-length/2,  width/2],
        [ length/2,  width/2],   # close the box
    ])
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)],
    ])
    rotated = corners @ R.T + center[:2]
    ax.plot(rotated[:, 0], rotated[:, 1], color=color, linewidth=linewidth, zorder=zorder)


def compute_average_car_size(nusc):
    sizes = [
        ann["size"]
        for ann in nusc.sample_annotation
        if "vehicle.car" in ann["category_name"]
    ]
    return np.array(sizes).mean(axis=0)


def visualize_motion_cone(nusc, speed_min_kmh=50, speed_max_kmh=60):
    avg_size = compute_average_car_size(nusc)
    print(f"Average car size (w, l, h): {avg_size}")

    next_positions_local = []
    speeds_found = []

    for sample in tqdm(nusc.sample, desc="Scanning samples"):
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)

            if "vehicle.car" not in ann["category_name"]:
                continue

            if ann["next"] == "":
                continue

            next_ann = nusc.get("sample_annotation", ann["next"])

            # ── compute speed from actual displacement and actual dt ───────
            p0 = np.array(ann["translation"][:2])
            p1 = np.array(next_ann["translation"][:2])

            t0 = get_ann_timestamp(nusc, ann)           # microseconds
            t1 = get_ann_timestamp(nusc, next_ann)
            dt = (t1 - t0) / 1e6                        # convert to seconds

            if dt <= 0:
                continue

            dist = np.linalg.norm(p1 - p0)
            speed_ms = dist / dt
            speed_kmh = speed_ms * 3.6

            if not (speed_min_kmh <= speed_kmh <= speed_max_kmh):
                continue

            # ── transform into car-local frame ────────────────────────────
            yaw = get_yaw(ann)
            R = np.array([
                [ np.cos(yaw), np.sin(yaw)],
                [-np.sin(yaw), np.cos(yaw)],
            ])
            local = R @ (p1 - p0)

            next_positions_local.append(local)
            speeds_found.append(speed_kmh)

    n = len(next_positions_local)
    print(f"\nCars in {speed_min_kmh}–{speed_max_kmh} km/h with a next annotation: {n}")
    if n == 0:
        print("Nothing to plot.")
        return

    pts = np.array(next_positions_local)

    distances = np.linalg.norm(pts, axis=1)
    print(f"Distance from centre — min: {distances.min():.2f} m, "
          f"max: {distances.max():.2f} m, "
          f"mean: {distances.mean():.2f} m")

    # ── plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))

    # scatter of all "next" centers in local frame
    sc = ax.scatter(
        pts[:, 0], pts[:, 1],
        c=speeds_found, cmap="plasma",
        s=60, alpha=0.75, zorder=2,
        label=f"Next-frame centers (n={n})"
    )
    plt.colorbar(sc, ax=ax, label="Speed (km/h)")

    # reference car box at origin, pointing in the +X direction
    draw_car_box(ax, center=np.array([0, 0, 0]), size=avg_size, yaw=0,
                 color="red", linewidth=2.5, zorder=4)
    ax.scatter(0, 0, s=160, color="red", zorder=5, label="Reference car")

    # forward-direction arrow
    ax.annotate("", xy=(avg_size[1], 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))

    ax.set_aspect("equal")
    ax.set_title(
        f"Motion Cone — next-frame centers in car-local frame\n"
        f"Speed range: {speed_min_kmh}–{speed_max_kmh} km/h  |  NuScenes mini"
    )
    ax.set_xlabel("X  (forward, m)")
    ax.set_ylabel("Y  (left, m)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_ylim(-5, 5)
    ax.set_xlim(-5, 25)

    # ── compute motion radius ───────────────────────────────
    speed_avg_kmh = (speed_min_kmh + speed_max_kmh) / 2
    print(f"\nAverage speed in range: {speed_avg_kmh:.2f} km/h")
    speed_ms = speed_avg_kmh / 3.6
    print(f"Average speed: {speed_ms:.2f} m/s")

    dt = 0.5  # seconds between frames
    a = 6.0   # m/s^2 acceleration assumption

    radius = speed_ms * dt + 0.5 * a * dt**2

    print(f"Motion radius: {radius:.2f} m")

    # ── draw circle ─────────────────────────────────────────
    circle = plt.Circle(
        (0, 0),
        radius,
        color="blue",
        fill=False,
        linestyle="--",
        linewidth=2,
        label=f"Reachable area (~{radius:.2f} m)"
    )
    ax.add_patch(circle)


    plt.tight_layout()
    plt.savefig("mini_30_35.png")  # uloží do súboru

    plt.show()


# ── entry point ────────────────────────────────────────────────────────────
#nusc = NuScenes(version="v1.0-mini", dataroot="../../nuscenes")
nusc = NuScenes(version="v1.0-trainval", dataroot="../../Nuscene")
visualize_motion_cone(nusc, speed_min_kmh=50, speed_max_kmh=51)