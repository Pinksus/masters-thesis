from box_utils import BoxUtils
import numpy as np
from scipy.optimize import minimize
import copy
from pyquaternion import Quaternion
from bev_fit_utils import BEVFitUtils
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
import numpy as np
import time
import os


class BEVPlotter:

    @staticmethod
    def box_to_bev_polygon(box):
        """
        Robustný BEV polygon:
        - vyberie 4 najnižšie rohy (Z)
        - zoradí ich pomocou ConvexHull
        - invariantné na yaw + π
        """
        corners = box.corners().T          # (8, 3)

        # vyber 4 rohy s najmenším Z
        idx = np.argsort(corners[:, 2])[:4]
        bev = corners[idx][:, :2]          # (4, 2)

        # zoradenie CCW
        hull = ConvexHull(bev)
        ordered = bev[hull.vertices]

        return Polygon(ordered)

    @staticmethod
    def bev_iou(box1, box2):
        p1 = BEVPlotter.box_to_bev_polygon(box1)
        p2 = BEVPlotter.box_to_bev_polygon(box2)

        if not p1.is_valid or not p2.is_valid:
            return 0.0

        inter = p1.intersection(p2).area
        union = p1.union(p2).area

        return 0.0 if union == 0 else inter / union

    @staticmethod
    def fit_box_to_points(prev_box, points, scale=1.0):
        """
        Fit a box to a set of points in BEV (X, Y).
        Returns:
            best_box: NuScenes Box object posunutý a otočený optimálne
            result: scipy OptimizeResult
        """

        # --- Pripravíme body ---
        pts = points[:, :2]  # X, Y

        # --- Funkcia, ktorú chceme minimalizovať ---
        def loss_fn(params):
            """
            params = [dx, dy, dtheta]
            dx, dy: translation
            dtheta: rotation in radians
            """
            dx, dy, dtheta = params

            # vytvoríme skopírovaný box
            box = copy.deepcopy(prev_box)
            box.center[:2] += np.array([dx, dy])
            box.orientation = prev_box.orientation * Quaternion(axis=[0,0,1], angle=dtheta)

            # získame hranice boxu v BEV
            corners = box.corners()[:2, :].T  # 8x2
            min_xy = np.min(corners, axis=0)
            max_xy = np.max(corners, axis=0)

            # penalizácia bodov mimo boxu
            dx = np.maximum(min_xy[0] - pts[:,0], 0) + np.maximum(pts[:,0] - max_xy[0], 0)
            dy = np.maximum(min_xy[1] - pts[:,1], 0) + np.maximum(pts[:,1] - max_xy[1], 0)

            return np.mean(dx**2 + dy**2)  # priemerná štvorcová vzdialenosť

        # --- inicializačné parametre ---
        x0 = [0.0, 0.0, 0.0]  # žiadny posun, žiadna rotácia

        result = minimize(loss_fn, x0, method='Powell')

        # --- vytvoríme optimalizovaný box ---
        best_box = copy.deepcopy(prev_box)
        best_box.center[:2] += result.x[:2]
        best_box.orientation = prev_box.orientation * Quaternion(axis=[0,0,1], angle=result.x[2])

        # ak scale != 1
        if scale != 1.0:
            best_box.wlh *= scale

        return best_box, result

    @staticmethod
    def plot(ax, points, prv_box, prev_box, box_curr, scaled_box, distance):
        start_time = time.time()

        # Crop points inside previous box
        inside_xy = BoxUtils.crop_points_to_prev_box(points, prev_box)

        # Filter points by height
        z_bottom = prev_box.center[2] - prev_box.wlh[2] / 2
        HEIGHT_OFFSET = 0.30
        inside_pts = inside_xy[inside_xy[:, 2] >= z_bottom + HEIGHT_OFFSET]

        # If not enough points, show warning and exit
        if inside_pts.shape[0] < 3:
            ax.text(prev_box.center[0], prev_box.center[1], 'NO POINTS', color='red')
            return

        # Scatter plot of points
        ax.scatter(inside_pts[:, 0], inside_pts[:, 1], s=2, label='inside points')

        points_2d = inside_pts[:, :2]

        # Compute minimum area rectangle around points
        from scipy.spatial import ConvexHull
        import numpy as np

        def minimum_area_rectangle(points_2d):
            hull = ConvexHull(points_2d)
            hull_points = points_2d[hull.vertices]

            min_area = np.inf
            best_rect = None

            for i in range(len(hull_points)):
                p1 = hull_points[i]
                p2 = hull_points[(i + 1) % len(hull_points)]
                edge_dir = p2 - p1
                edge_angle = -np.arctan2(edge_dir[1], edge_dir[0])

                R = np.array([
                    [np.cos(edge_angle), -np.sin(edge_angle)],
                    [np.sin(edge_angle),  np.cos(edge_angle)]
                ])

                rot_points = hull_points @ R.T
                min_xy = np.min(rot_points, axis=0)
                max_xy = np.max(rot_points, axis=0)
                area = np.prod(max_xy - min_xy)

                if area < min_area:
                    min_area = area
                    center_rot = (min_xy + max_xy) / 2
                    best_rect = {
                        'center': R.T @ center_rot,
                        'size': max_xy - min_xy,
                        'angle': -edge_angle
                    }

            return best_rect

        rect = minimum_area_rectangle(points_2d)
        hull_center = rect['center']
        hull_angle = rect['angle']

        prev_center = prev_box.center[:2]
        center_dist = np.linalg.norm(hull_center - prev_center)

        # Decide final center using threshold
        CENTER_THRESHOLD = 0.5
        if center_dist <= CENTER_THRESHOLD:
            final_center = hull_center
        else:
            final_center = prev_center

        # Plot centers
        ax.scatter(prev_center[0], prev_center[1], c='blue', s=80, marker='x', label='prev_box center')
        ax.scatter(hull_center[0], hull_center[1], c='magenta', s=80, marker='o', label='convex hull center')
        ax.scatter(final_center[0], final_center[1], c='cyan', s=80, marker='*', label='final gated center')

        # Initialize aligned box
        import copy
        from pyquaternion import Quaternion

        init_box = copy.deepcopy(prev_box)
        init_box.center[:2] = final_center
        init_box.orientation = Quaternion(axis=[0, 0, 1], angle=hull_angle)

        aligned_box, res = BEVFitUtils.fit_box(init_box, inside_pts)

        # Visualization of aligned box
        order = [3, 7, 6, 2, 3]
        corners = aligned_box.corners().T
        ax.plot(corners[order, 0], corners[order, 1], color='green', linewidth=2, label='aligned prev_box')

        for box, style in [
            (scaled_box, dict(color='red', label='scaled')),
            (prev_box, dict(color='blue', linestyle='--', label='original')),
            (box_curr, dict(color='orange', label='ground_truth'))
        ]:
            c = box.corners().T
            ax.plot(c[order, 0], c[order, 1], **style)

        ax.set_aspect('equal')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))




