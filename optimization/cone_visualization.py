import os
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from matplotlib.patches import Ellipse
from tqdm import tqdm


# ───────────────── constants ─────────────────

TURN_RADIUS_GEOMETRIC = 3.8   # Smart ForTwo — mechanický limit [m]
MU   = 0.8                    # koeficient bočného trenia pneumatík
G    = 9.81                   # gravitačné zrýchlenie [m/s²]
DT   = 0.5                    # časový krok [s]


# ───────────────── helpers ─────────────────

def effective_turn_radius(speed_ms,
                          r_geom=TURN_RADIUS_GEOMETRIC,
                          mu=MU, g=G):
    """
    Skutočný minimálny polomer otáčania pri danej rýchlosti.

    Sú dva limity:
      1. Geometrický (volant nadoraz):  r_geom = 3.8 m
      2. Trecia sila pneumatík:         r_friction = v² / (μ·g)

    Efektívny polomer = max z oboch — pri nízkych rýchlostiach
    dominuje geometria, pri vysokých dominuje trecia sila.

    Crossover ≈ sqrt(r_geom · μ · g) = sqrt(3.8·0.8·9.81) ≈ 5.5 m/s ≈ 20 km/h
    """
    r_friction = speed_ms ** 2 / (mu * g)
    return max(r_geom, r_friction)

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

def draw_turning_cone(ax, arc_length, r_eff):
    """
    Nakreslí hranicu kužeľa otáčania.

    arc_length = max. dráha za DT sekúnd  (= v·dt + 0.5·a·dt²)
    r_eff      = efektívny polomer otáčania pri danej rýchlosti

    Maximálny uhol:   theta_max = min(arc_length / r_eff, pi/2)
    Maximálne x:      x_max = r_eff · sin(theta_max)
    Bočná hranica:    y_max(x) = r_eff - sqrt(r_eff² - x²)

    Pri vysokej rýchlosti je r_eff >> r_geom → kužeľ je úzky.
    Pri nízkej rýchlosti je r_eff = r_geom  → kužeľ je najširší.
    """
    theta_max = min(arc_length / r_eff, np.pi / 2)
    x_max = r_eff * np.sin(theta_max)

    x_vals = np.linspace(0, x_max, 300)
    y_upper = r_eff - np.sqrt(np.maximum(r_eff**2 - x_vals**2, 0))
    y_lower = -y_upper

    label = (f"Turning cone  r_eff={r_eff:.1f}m  "
             f"θ_max={np.degrees(theta_max):.1f}°")

    ax.plot(x_vals, y_upper, color="green", linewidth=2, linestyle="-", label=label)
    ax.plot(x_vals, y_lower, color="green", linewidth=2, linestyle="-")
    ax.plot([x_max, x_max], [y_lower[-1], y_upper[-1]],
            color="green", linewidth=2, linestyle="-")

def compute_average_car_size(nusc):
    sizes = [
        ann["size"]
        for ann in nusc.sample_annotation
        if "vehicle.car" in ann["category_name"]
    ]
    return np.array(sizes).mean(axis=0)


# ───────────────── outlier checks ─────────────────

def is_outside_ellipse(point, radius, width_scale):
    x, y = point
    a = radius
    b = radius * width_scale
    return (x / a) ** 2 + (y / b) ** 2 > 1.0

def is_outside_turning_cone(point, arc_length, r_eff):
    """
    Vráti True ak bod leží mimo kužeľa otáčania.

    r_eff = effective_turn_radius(speed_ms) — závisí od rýchlosti:
      - nízka rýchlosť (~5 km/h):  r_eff ≈ 3.8m  → kužeľ relatívne široký
      - vysoká rýchlosť (60 km/h): r_eff ≈ 35m   → kužeľ veľmi úzky
        (auto nemôže prudko zatočiť bez šmyku)
    """
    x, y = point

    if x < 0:
        return True

    theta_max = min(arc_length / r_eff, np.pi / 2)
    x_max = r_eff * np.sin(theta_max)

    if x > x_max:
        return True

    if x >= r_eff:
        y_limit = r_eff
    else:
        y_limit = r_eff - np.sqrt(r_eff**2 - x**2)

    return abs(y) > y_limit


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


def plot_bins(bins, avg_size, bin_size=2, output_dir="examples_6"):
    os.makedirs(output_dir, exist_ok=True)

    outlier_counts = {}

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

        # ── parametre ──
        a = accel_by_speed(speed_avg_kmh)
        arc_length  = speed_ms * DT + 0.5 * a * DT**2
        width_scale = width_scale_by_speed(speed_avg_kmh)
        r_eff       = effective_turn_radius(speed_ms)

        # ── rozdeľ body ──
        outside_mask = np.array([
            is_outside_ellipse(p, arc_length, width_scale) or
            is_outside_turning_cone(p, arc_length, r_eff)
            for p in pts
        ])
        inliers  = pts[~outside_mask]
        outliers = pts[ outside_mask]

        bin_label = f"{v_min}-{v_max} km/h"
        outlier_counts[bin_label] = len(outliers)

        # ── plot ──
        fig, ax = plt.subplots(figsize=(10, 10))

        if len(inliers) > 0:
            inlier_speeds = np.array(speeds)[~outside_mask]
            sc = ax.scatter(
                inliers[:, 0], inliers[:, 1],
                c=inlier_speeds, cmap="plasma",
                s=60, alpha=0.75, label="Inliers"
            )
            plt.colorbar(sc, ax=ax, label="Speed (km/h)")

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

        # elipsa (modrá prerušovaná)
        ellipse = Ellipse(
            (0, 0),
            width=arc_length * 2,
            height=arc_length * width_scale * 2,
            edgecolor="blue",
            fill=False,
            linestyle="--",
            linewidth=2,
            label="Ellipse boundary"
        )
        ax.add_patch(ellipse)

        # kužeľ otáčania (zelený) — závisí od r_eff(v)
        draw_turning_cone(ax, arc_length, r_eff)

        total = len(pts)
        pct = 100 * len(outliers) / total if total > 0 else 0
        theta_max_deg = np.degrees(min(arc_length / r_eff, np.pi / 2))

        ax.set_title(
            f"{v_min}-{v_max} km/h  |  "
            f"arc={arc_length:.2f}m  r_eff={r_eff:.1f}m  θ_max={theta_max_deg:.1f}°  |  "
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