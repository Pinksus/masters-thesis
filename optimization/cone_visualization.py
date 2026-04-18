import os
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from matplotlib.patches import Ellipse
from tqdm import tqdm


# ───────────────── helpers ─────────────────

def width_scale_by_speed(speed_avg_kmh, min_speed=0, max_speed=70):
    min_width = 0.4
    scale = 1.0 - (1.0 - min_width) * (speed_avg_kmh - min_speed) / (max_speed - min_speed)
    scale = np.clip(scale, min_width, 1.0)
    return scale

def accel_by_speed(speed_avg_kmh):
    a_min = 4.0
    a_max = 12.0
    v_c = 19.0
    b = 0.35
    a = a_min + (a_max - a_min) / (1 + np.exp(-b * (speed_avg_kmh - v_c)))
    return a

def get_yaw(ann):
    q = Quaternion(ann["rotation"])
    return q.yaw_pitch_roll[0]

def get_ann_timestamp(nusc, ann):
    sample = nusc.get("sample", ann["sample_token"])
    return sample["timestamp"]

def draw_car_box(ax, center, size, yaw, color="red", linewidth=2):
    width, length = size[0], size[1]
    corners = np.array([
        [ length/2,  width/2],
        [ length/2, -width/2],
        [-length/2, -width/2],
        [-length/2,  width/2],
        [ length/2,  width/2],
    ])
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)],
    ])
    rotated = corners @ R.T + center[:2]
    ax.plot(rotated[:, 0], rotated[:, 1], color=color, linewidth=linewidth)

def compute_average_car_size(nusc):
    sizes = [
        ann["size"]
        for ann in nusc.sample_annotation
        if "vehicle.car" in ann["category_name"]
    ]
    return np.array(sizes).mean(axis=0)


# ───────────────── outlier check ─────────────────

def is_outside_ellipse(point, radius, width_scale):
    """
    Vráti True ak bod leží mimo elipsy.
    Elipsa je vycentrovaná v (0,0):
      - polos x (forward) = radius
      - polos y (lateral) = radius * width_scale
    """
    x, y = point
    a = radius          # polos v smere jazdy (x)
    b = radius * width_scale   # polos do strany (y)
    return (x / a) ** 2 + (y / b) ** 2 > 1.0


# ───────────────── MAIN LOGIC ─────────────────

def collect_data(nusc, max_speed=70, bin_size=2):
    bins = {}
    num_bins = int(max_speed / bin_size)

    for i in range(num_bins):
        bins[i] = {"pts": [], "speeds": []}

    for sample in tqdm(nusc.sample, desc="Collecting data"):
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)

            if "vehicle.car" not in ann["category_name"]:
                continue
            if ann["next"] == "":
                continue

            next_ann = nusc.get("sample_annotation", ann["next"])

            p0 = np.array(ann["translation"][:2])
            p1 = np.array(next_ann["translation"][:2])

            t0 = get_ann_timestamp(nusc, ann)
            t1 = get_ann_timestamp(nusc, next_ann)
            dt = (t1 - t0) / 1e6

            if dt <= 0:
                continue

            dist = np.linalg.norm(p1 - p0)
            speed_ms = dist / dt
            speed_kmh = speed_ms * 3.6

            if speed_kmh >= max_speed:
                continue

            bin_idx = int(speed_kmh // bin_size)

            yaw = get_yaw(ann)
            R = np.array([
                [ np.cos(yaw), np.sin(yaw)],
                [-np.sin(yaw), np.cos(yaw)],
            ])
            local = R @ (p1 - p0)

            bins[bin_idx]["pts"].append(local)
            bins[bin_idx]["speeds"].append(speed_kmh)

    return bins


def plot_bins(bins, avg_size, bin_size=2, output_dir="examples_5"):
    os.makedirs(output_dir, exist_ok=True)

    outlier_counts = {}  # { "0-2 km/h": count, ... }

    for bin_idx, data in bins.items():
        pts = data["pts"]
        speeds = data["speeds"]

        if len(pts) == 0:
            continue

        pts = np.array(pts)

        v_min = bin_idx * bin_size
        v_max = v_min + bin_size
        speed_avg_kmh = (v_min + v_max) / 2
        speed_ms = speed_avg_kmh / 3.6

        # ── elipsa parametre ──
        dt = 0.5
        a = accel_by_speed(speed_avg_kmh)
        radius = speed_ms * dt + 0.5 * a * dt**2
        width_scale = width_scale_by_speed(speed_avg_kmh, min_speed=0, max_speed=70)

        # ── rozdeľ body na inliers / outliers ──
        outside_mask = np.array([
            is_outside_ellipse(p, radius, width_scale) for p in pts
        ])
        inliers  = pts[~outside_mask]
        outliers = pts[ outside_mask]

        bin_label = f"{v_min}-{v_max} km/h"
        outlier_counts[bin_label] = len(outliers)

        # ── plot ──
        fig, ax = plt.subplots(figsize=(10, 10))

        # inliers — farba podľa rýchlosti
        if len(inliers) > 0:
            inlier_speeds = np.array(speeds)[~outside_mask]
            sc = ax.scatter(
                inliers[:, 0], inliers[:, 1],
                c=inlier_speeds, cmap="plasma",
                s=60, alpha=0.75, label="Inliers"
            )
            plt.colorbar(sc, ax=ax, label="Speed (km/h)")

        # outliers — červená bodka
        if len(outliers) > 0:
            ax.scatter(
                outliers[:, 0], outliers[:, 1],
                color="red", s=60, alpha=0.85,
                zorder=5, label=f"Outliers ({len(outliers)})"
            )

        draw_car_box(ax, center=np.array([0, 0, 0]), size=avg_size, yaw=0)
        ax.scatter(0, 0, s=150, color="red", zorder=6)

        ax.annotate("", xy=(avg_size[1], 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))

        ellipse = Ellipse(
            (0, 0),
            width=radius * 2,
            height=radius * width_scale * 2,
            edgecolor='blue',
            fill=False,
            linestyle='--',
            linewidth=2
        )
        ax.add_patch(ellipse)

        total = len(pts)
        pct = 100 * len(outliers) / total if total > 0 else 0
        ax.set_title(
            f"{v_min}-{v_max} km/h  |  "
            f"outliers: {len(outliers)}/{total} ({pct:.1f}%)"
        )
        ax.set_xlim(-5, 25)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/test_{v_min}-{v_max}.png")
        plt.close(fig)

    return outlier_counts


# ───────────────── ENTRY POINT ─────────────────

nusc = NuScenes(version="v1.0-trainval", dataroot="../../Nuscene")

print("Computing average car size...")
avg_size = compute_average_car_size(nusc)

print("Collecting data (one pass)...")
bins = collect_data(nusc)

print("Plotting...")
outlier_counts = plot_bins(bins, avg_size)

print("\n── Outlier summary ──")
total_outliers = 0
for label, count in outlier_counts.items():
    print(f"  {label}: {count} outliers")
    total_outliers += count
print(f"  TOTAL: {total_outliers} outliers")

print("\nDone.")