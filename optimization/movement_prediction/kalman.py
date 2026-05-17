import matplotlib.pyplot as plt
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from nuscenes import NuScenes

nusc = NuScenes(version="v1.0-mini", dataroot="../../../nuscenes")


# =====================================================
# EKF  —  CTRV model
#
# Stav:    x = [px, py, v, θ, ω]
#   px, py  — poloha v globálnom súradnicovom systéme
#   v       — rýchlosť (skalár)
#   θ       — yaw uhol (smer jazdy)
#   ω       — yaw rate (ako rýchlo sa otáča)
#
# Meranie: z = [px, py, θ]
#
# Prečo CTRV a nie CV?
#   CV predikuje vždy PRIAMKU — nezná koncept zatáčania.
#   CTRV modeluje ω → vie predikovať oblúky a krivky.
# =====================================================

class CTRV_EKF:

    def __init__(self, sigma_a=0.5, sigma_omega=0.2,
                 sigma_pos=0.3, sigma_yaw=0.05):
        self.sigma_a     = sigma_a
        self.sigma_omega = sigma_omega

        self.R = np.diag([sigma_pos**2, sigma_pos**2, sigma_yaw**2])

        self.H = np.zeros((3, 5))
        self.H[0, 0] = 1.0   # px
        self.H[1, 1] = 1.0   # py
        self.H[2, 3] = 1.0   # θ

        self.x = None
        self.P = None

    @staticmethod
    def _normalize_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def init(self, px, py, v, theta, omega):
        self.x = np.array([px, py, v, theta, omega], dtype=float)
        self.P = np.diag([2.0, 2.0, 2.0, 0.5, 0.5])

    def _f(self, x, dt):
        px, py, v, theta, omega = x
        if abs(omega) > 1e-4:
            px_new = px + (v / omega) * ( np.sin(theta + omega * dt) - np.sin(theta))
            py_new = py + (v / omega) * (-np.cos(theta + omega * dt) + np.cos(theta))
        else:
            px_new = px + v * np.cos(theta) * dt
            py_new = py + v * np.sin(theta) * dt
        theta_new = self._normalize_angle(theta + omega * dt)
        return np.array([px_new, py_new, v, theta_new, omega])

    def _F_jacobian(self, x, dt):
        _, _, v, theta, omega = x
        F = np.eye(5)
        if abs(omega) > 1e-4:
            s0 = np.sin(theta);          c0 = np.cos(theta)
            s1 = np.sin(theta+omega*dt); c1 = np.cos(theta+omega*dt)
            v_ow = v / omega;            v_o2 = v / omega**2

            F[0, 2] =  (s1 - s0) / omega
            F[0, 3] =  v_ow * (c1 - c0)
            F[0, 4] =  v_ow * c1 * dt - v_o2 * (s1 - s0)

            F[1, 2] =  (-c1 + c0) / omega
            F[1, 3] =  v_ow * (s1 - s0)
            F[1, 4] =  v_ow * s1 * dt - v_o2 * (-c1 + c0)
        else:
            c0 = np.cos(theta); s0 = np.sin(theta)
            F[0, 2] =  c0 * dt;      F[0, 3] = -v * s0 * dt
            F[1, 2] =  s0 * dt;      F[1, 3] =  v * c0 * dt

        F[3, 4] = dt
        return F

    def _Q(self, dt):
        dt2 = dt**2; dt3 = dt**3; dt4 = dt**4
        Qa = self.sigma_a**2 * np.array([
            [dt4/4, 0,     dt3/2, 0,     0    ],
            [0,     dt4/4, 0,     0,     0    ],
            [dt3/2, 0,     dt2,   0,     0    ],
            [0,     0,     0,     0,     0    ],
            [0,     0,     0,     0,     0    ],
        ])
        Qo = self.sigma_omega**2 * np.array([
            [0, 0, 0, 0,      0    ],
            [0, 0, 0, 0,      0    ],
            [0, 0, 0, 0,      0    ],
            [0, 0, 0, dt4/4,  dt3/2],
            [0, 0, 0, dt3/2,  dt2  ],
        ])
        return Qa + Qo

    def predict(self, dt):
        Fj     = self._F_jacobian(self.x, dt)
        self.x = self._f(self.x, dt)
        self.P = Fj @ self.P @ Fj.T + self._Q(dt)

    def update(self, z):
        z    = np.array(z, dtype=float)
        y    = z - self.H @ self.x
        y[2] = self._normalize_angle(y[2])
        S    = self.H @ self.P @ self.H.T + self.R
        K    = self.P @ self.H.T @ np.linalg.inv(S)
        self.x    = self.x + K @ y
        self.x[3] = self._normalize_angle(self.x[3])
        self.P    = (np.eye(5) - K @ self.H) @ self.P

    def predict_n(self, n, dt, a_est=0.0):
        """
        Predikuje n krokov dopredu.
        a_est — odhadnuté zrýchlenie [m/s/krok], získané z lineárnej
                regresie rýchlostí v histórii. Kladné = zrýchľuje,
                záporné = spomaľuje. Bez tohto parametru ostáva v konštantná.
        """
        x, P = self.x.copy(), self.P.copy()
        preds = []
        for _ in range(n):
            x[2] = max(0.0, x[2] + a_est * dt)   # v += a*dt, min 0
            Fj = self._F_jacobian(x, dt)
            x  = self._f(x, dt)
            P  = Fj @ P @ Fj.T + self._Q(dt)
            preds.append(x[:2].copy())
        return np.array(preds)


# =====================================================
# 1. CURRENT SAMPLE + CURRENT ANNOTATION
# =====================================================
sample    = nusc.sample[56]
ann_token = sample["anns"][12]
ann       = nusc.get("sample_annotation", ann_token)


# =====================================================
# 2. LOAD TRAJECTORY  (current + 5 previous)
# =====================================================
current_ann = ann
boxes      = []
centers    = []
yaws       = []
timestamps = []

for i in range(6):
    box = nusc.get_box(current_ann["token"])
    boxes.append(box)
    centers.append(box.center.copy())

    q = Quaternion(matrix=box.rotation_matrix)
    yaws.append(q.yaw_pitch_roll[0])

    s_token = current_ann["sample_token"]
    ts = nusc.get("sample", s_token)["timestamp"] / 1e6
    timestamps.append(ts)

    prev_token = current_ann["prev"]
    if prev_token == "":
        break
    current_ann = nusc.get("sample_annotation", prev_token)

# oldest → newest
boxes      = boxes[::-1]
centers    = centers[::-1]
yaws       = yaws[::-1]
timestamps = timestamps[::-1]


# =====================================================
# 2b. FUTURE GROUND TRUTH  (next anotácie = skutočná trasa)
#     zbierame max 8 krokov dopredu, rovnako ako predikcia
# =====================================================
future_gt  = []
future_ann = ann   # current annotation (pred reversom sme začínali tu)

for _ in range(8):
    next_token = future_ann["next"]
    if next_token == "":
        break
    future_ann = nusc.get("sample_annotation", next_token)
    future_gt.append(nusc.get_box(future_ann["token"]).center[:2].copy())

future_gt = np.array(future_gt) if future_gt else None


# =====================================================
# 3. CTRV EKF  — inicializácia + trénuj na histórii
# =====================================================
kf = CTRV_EKF(sigma_a=0.5, sigma_omega=0.2,
               sigma_pos=0.3, sigma_yaw=0.05)

last_dt = 0.5

for i in range(len(centers)):
    px, py = centers[i][0], centers[i][1]
    theta  = yaws[i]

    if i == 0:
        if len(centers) > 1:
            dt0  = timestamps[1] - timestamps[0]
            dx   = centers[1][0] - centers[0][0]
            dy   = centers[1][1] - centers[0][1]
            v0   = np.hypot(dx, dy) / max(dt0, 1e-6)
            d_th = CTRV_EKF._normalize_angle(yaws[1] - yaws[0])
            om0  = d_th / max(dt0, 1e-6)
        else:
            v0, om0 = 0.0, 0.0
        kf.init(px, py, v0, theta, om0)
    else:
        dt      = timestamps[i] - timestamps[i - 1]
        last_dt = dt
        kf.predict(dt)
        kf.update([px, py, theta])

# ── Korekcia stavu po tréningu ─────────────────────────────────
# Prepíšeme v, θ aj ω priamo z nameraných bodov — filter ich mal
# skreslené, lebo v a ω nie sú priamo merané (len px, py, θ).
if len(centers) >= 2:
    dt_last = timestamps[-1] - timestamps[-2]
    dx      = centers[-1][0] - centers[-2][0]
    dy      = centers[-1][1] - centers[-2][1]

    # v — rýchlosť z posledného kroku histórie
    kf.x[2] = np.hypot(dx, dy) / max(dt_last, 1e-6)

    # θ — skutočný smer pohybu (nie orientácia boxu)
    kf.x[3] = np.arctan2(dy, dx)

    # ω — zmena smeru pohybu za posledné dva kroky histórie
    if len(centers) >= 3:
        dt_prev  = timestamps[-2] - timestamps[-3]
        dx_prev  = centers[-2][0] - centers[-3][0]
        dy_prev  = centers[-2][1] - centers[-3][1]
        th_prev  = np.arctan2(dy_prev, dx_prev)
        th_curr  = np.arctan2(dy, dx)
        kf.x[4] = CTRV_EKF._normalize_angle(th_curr - th_prev) / max(dt_prev, 1e-6)
    else:
        kf.x[4] = 0.0

# ── Odhad zrýchlenia z histórie (lineárna regresia rýchlostí) ──
velocities = []
for i in range(1, len(centers)):
    dt_i = timestamps[i] - timestamps[i - 1]
    dx   = centers[i][0] - centers[i - 1][0]
    dy   = centers[i][1] - centers[i - 1][1]
    velocities.append(np.hypot(dx, dy) / max(dt_i, 1e-6))

if len(velocities) >= 2:
    t_v   = np.arange(len(velocities), dtype=float)
    # polyfit stupeň 1 → [sklon, intercept]; sklon = delta_v za krok
    a_est = np.polyfit(t_v, velocities, 1)[0]
    print(f"Odhadnuté zrýchlenie: {a_est:+.3f} m/s per krok  "
          f"({'zrýchľuje' if a_est > 0.05 else 'spomaľuje' if a_est < -0.05 else 'konštantná v'})")
else:
    a_est = 0.0
    print("Málo bodov na odhad zrýchlenia — predikcia s konštantnou v")

future_pts = kf.predict_n(8, last_dt, a_est=a_est)


# =====================================================
# 4. LOAD CURRENT LIDAR
# =====================================================
lidar_token = sample["data"]["LIDAR_TOP"]
pc          = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))

sd_record   = nusc.get("sample_data", lidar_token)
cs_record   = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])


# =====================================================
# 5. TRANSFORM POINT CLOUD -> GLOBAL FRAME
# =====================================================
pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
pc.translate(np.array(cs_record["translation"]))
pc.rotate(Quaternion(pose_record["rotation"]).rotation_matrix)
pc.translate(np.array(pose_record["translation"]))


# =====================================================
# 6. VISUALIZATION
# =====================================================
fig, ax = plt.subplots(figsize=(12, 12))

points = pc.points
ax.scatter(points[0, :], points[1, :], s=0.2, c="gray", alpha=0.5)

centers_arr = np.array(centers)[:, :2]
ax.plot(centers_arr[:, 0], centers_arr[:, 1], "o--", label="Trajectory", zorder=3)

for i, box in enumerate(boxes):
    color = ("r", "r", "r") if i == len(boxes) - 1 else ("b", "b", "b")
    box.render(ax, view=np.eye(4), colors=color)
    ax.text(box.center[0], box.center[1], f"{i}")

# ── Kružnica opísaná posledným 3 bodmi histórie ─────────
def circumscribed_circle(p1, p2, p3):
    ax2, ay2 = p1;  bx, by = p2;  cx2, cy2 = p3
    D = 2 * (ax2*(by-cy2) + bx*(cy2-ay2) + cx2*(ay2-by))
    if abs(D) < 1e-10:
        return None, None, None
    ux = ((ax2**2+ay2**2)*(by-cy2) + (bx**2+by**2)*(cy2-ay2) + (cx2**2+cy2**2)*(ay2-by)) / D
    uy = ((ax2**2+ay2**2)*(cx2-bx) + (bx**2+by**2)*(ax2-cx2) + (cx2**2+cy2**2)*(bx-ax2)) / D
    r  = np.hypot(ax2-ux, ay2-uy)
    return ux, uy, r

if len(centers_arr) >= 3:
    p1, p2, p3 = centers_arr[-3], centers_arr[-2], centers_arr[-1]
    ccx, ccy, ccr = circumscribed_circle(p1, p2, p3)

    if ccx is not None:
        v2    = p2 - np.array([ccx, ccy])
        v3    = p3 - np.array([ccx, ccy])
        cross = v2[0]*v3[1] - v2[1]*v3[0]
        ccw   = cross > 0

        a_curr = np.arctan2(p3[1]-ccy, p3[0]-ccx)

        if ccw:
            angles = np.linspace(a_curr, a_curr + np.radians(120), 80)
        else:
            angles = np.linspace(a_curr, a_curr - np.radians(120), 80)

        arc_x = ccx + ccr * np.cos(angles)
        arc_y = ccy + ccr * np.sin(angles)

        theta_full = np.linspace(0, 2*np.pi, 300)
        ax.plot(ccx + ccr*np.cos(theta_full),
                ccy + ccr*np.sin(theta_full),
                "--", color="mediumpurple", linewidth=0.8,
                alpha=0.4, zorder=3)

        ax.plot(arc_x, arc_y,
                "-", color="mediumpurple", linewidth=2.0,
                label=f"Kružnica (r={ccr:.1f} m)", zorder=4)

        ax.plot(ccx, ccy, "+", color="mediumpurple", markersize=10, zorder=4)

        print(f"Kružnica: stred=({ccx:.2f}, {ccy:.2f}), r={ccr:.2f} m, "
              f"smer={'CCW' if ccw else 'CW'}")

# ── CTRV predikcia ────────────────────────────────────
connector = np.vstack([centers_arr[-1], future_pts[0]])
ax.plot(connector[:, 0], connector[:, 1], color="lime", linewidth=1.5, zorder=4)
ax.plot(
    future_pts[:, 0], future_pts[:, 1],
    "o--", color="lime", linewidth=1.5, markersize=6,
    label="CTRV predikcia", zorder=4,
)
for i, pt in enumerate(future_pts):
    ax.text(pt[0] + 0.1, pt[1] + 0.1, f"+{i+1}",
            color="lime", fontsize=8, fontweight="bold")

# ── Ground truth budúcnosť ────────────────────────────
if future_gt is not None and len(future_gt) > 0:
    connector_gt = np.vstack([centers_arr[-1], future_gt[0]])
    ax.plot(connector_gt[:, 0], connector_gt[:, 1],
            color="orange", linewidth=1.5, zorder=4)
    ax.plot(
        future_gt[:, 0], future_gt[:, 1],
        "o--", color="orange", linewidth=1.5, markersize=6,
        label="GT (skutočná trasa)", zorder=4,
    )
    for i, pt in enumerate(future_gt):
        ax.text(pt[0] + 0.1, pt[1] - 0.4, f"gt+{i+1}",
                color="orange", fontsize=8)

ax.set_aspect("equal")
ax.set_xlim(630, 680)
ax.set_ylim(1595, 1620)
ax.set_title("LiDAR + object history in global BEV  —  CTRV EKF predikcia")
ax.legend()
plt.show()