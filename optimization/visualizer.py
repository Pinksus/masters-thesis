import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, Box
import numpy as np
from pyquaternion import Quaternion
import copy
from nuscenes.utils.geometry_utils import view_points
from box_utils import BoxUtils

class Visualizer:

    @staticmethod
    def render_annotation(nusc,
                      anntoken: str,
                      stepback: int = 1,
                      margin: float = 10,
                      view: np.ndarray = np.eye(4)):

        # ---- Load current annotation ----
        ann_record = nusc.get('sample_annotation', anntoken)
        sample_record = nusc.get('sample', ann_record['sample_token'])
        instance_token = ann_record['instance_token']

        assert 'LIDAR_TOP' in sample_record['data']

        # --- Load LIDAR for current sample ---
        lidar_token = sample_record['data']['LIDAR_TOP']
        lidar_path, boxes, _ = nusc.get_sample_data(
            lidar_token,
            selected_anntokens=[anntoken]
        )

        # current box
        box_curr = boxes[0]
        print("Current box (local to current LIDAR):", box_curr.center)

        # ============================
        #   FIND PREVIOUS ANNOTATION
        # =============================
        frame_index = nusc.sample.index(sample_record)
        target_idx = frame_index - stepback
        print(f"Predosly sample bude: {target_idx}")

        prev_box_transformed = None

        if 0 <= target_idx < len(nusc.sample):
            prev_sample = nusc.sample[target_idx]

            # search annotation with same instance
            for a in prev_sample['anns']:
                ann_prev = nusc.get('sample_annotation', a)
                if ann_prev['instance_token'] == instance_token:

                    # --- Transform global coordinates of previous annotation to current LIDAR frame ---
                    p_global_prev = np.array(ann_prev['translation'])

                    # Load current ego pose + sensor calibration
                    sd_curr = nusc.get('sample_data', lidar_token)
                    cs_curr = nusc.get('calibrated_sensor', sd_curr['calibrated_sensor_token'])
                    ep_curr = nusc.get('ego_pose', sd_curr['ego_pose_token'])

                    # Global → current ego
                    p_ego_curr = Quaternion(ep_curr['rotation']).inverse.rotate(
                        p_global_prev - np.array(ep_curr['translation'])
                    )
                    # Current ego → current LIDAR
                    p_lidar_curr = Quaternion(cs_curr['rotation']).inverse.rotate(
                        p_ego_curr - np.array(cs_curr['translation'])
                    )

                    q_global = Quaternion(ann_prev['rotation'])
                    q_ego = Quaternion(ep_curr['rotation']).inverse * q_global
                    q_lidar = Quaternion(cs_curr['rotation']).inverse * q_ego

                    # Create Box object with same size and orientation as prev annotation
                    from nuscenes.utils.data_classes import Box
                    prev_box_transformed = Box(
                        center=p_lidar_curr,
                        size=ann_prev['size'],
                        orientation=q_lidar,
                        name=ann_prev['category_name'],
                        token=ann_prev['token']
                    )
                    break

        # ============================
        #       RENDERING
        # ============================
        pc = LidarPointCloud.from_file(lidar_path)
        pts = pc.points.T[:, :3]   # (x,y,z)


        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 10))
        fig.tight_layout()

        # pointcloud
        pc.render_height(ax_left, view=view)

        # CURRENT BOX — original color
        c_curr = np.array([0.0, 1.0, 0.0])
        box_curr.render(ax_left, view=view, colors=(c_curr, c_curr, c_curr))

        # PREVIOUS BOX — blue
        if prev_box_transformed is not None:
            c_prev = np.array([0.0, 0.0, 1.0])
            prev_box_transformed.render(ax_left, view=view, colors=(c_prev, c_prev, c_prev))
            print("Previous box transformed to current LIDAR frame:", prev_box_transformed.center)
        else:
            print("OBJECT DOES NOT EXIST in previous frame.")

        # Scaled box 1.5×
        scaled_prev = copy.deepcopy(prev_box_transformed)
        scaled_prev.wlh = prev_box_transformed.wlh * 1.5

        c_scaled = np.array([1.0, 0.0, 0.0])  # red
        scaled_prev.render(ax_left, view=view, colors=(c_scaled, c_scaled, c_scaled))

        # autoscale
        corners = view_points(box_curr.corners(), view, False)[:2, :]
        ax_left.set_xlim([np.min(corners[0]) - margin, np.max(corners[0]) + margin])
        ax_left.set_ylim([np.min(corners[1]) - margin, np.max(corners[1]) + margin])
        ax_left.set_aspect('equal')
        ax_left.set_title("LIDAR + Boxes")

        # ============================
        #       RIGHT PLOT (BEV)
        # ============================

        # Použijeme tvoju funkciu na výber bodov vo vnútri boxu
        inside_pts = BoxUtils.crop_points_to_prev_box(pts, prev_box_transformed)
        # scale=1.0 → pretože scaled_prev.wlh je už zväčšený o 1.5x

        # BEV scatter
        ax_right.scatter(inside_pts[:, 0], inside_pts[:, 1], s=2)

        order = [3,7,6,2,3]

        # scaled box (red)
        corn_scaled = scaled_prev.corners().T
        ax_right.plot(
            corn_scaled[order, 0],
            corn_scaled[order, 1],
            linewidth=2,
            color='red',
            label='scaled prev box'
        )

        # original prev box (blue)
        corn_prev = prev_box_transformed.corners().T
        ax_right.plot(
            corn_prev[order, 0],
            corn_prev[order, 1],
            linewidth=2,
            linestyle='--',
            color='blue',
            label='original prev box'
        )


        '''# inside_pts = pôvodne cropované body
        fitted_box, result = fit_box_to_points(prev_box_transformed, inside_pts)

        # nakreslíme fitted box v BEV
        corn_fitted = fitted_box.corners().T
        order = [3,7,6,2,3]
        ax_right.plot(
            corn_fitted[order,0],
            corn_fitted[order,1],
            color='green',
            linewidth=2,
            label='fitted prev box'
        )'''

        ax_right.legend()
        ax_right.set_aspect('equal')
        ax_right.set_title("BEV: original vs scaled prev box")

        plt.show()