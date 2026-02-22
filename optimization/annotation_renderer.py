import matplotlib.pyplot as plt
import numpy as np
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import LidarPointCloud
from bev_plotter import BEVPlotter


class AnnotationRenderer:
    """
    Pure visualization layer.
    """

    @staticmethod
    def render(result, margin=10, view=np.eye(4)):
        pts = result["points"]
        box_curr = result["box_curr"]
        prev_box = result["prev_box"]
        prev_box_moved = result["prev_box_moved"]
        scaled = result["scaled_box"]
        aligned_box = result["aligned_box"]  # ⭐ NEW
        distance = result["distance"]

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 10))

        # ---- left view ----
        pc = LidarPointCloud.from_file(result["lidar_path"])
        pc.render_height(ax_l, view=view)

        box_curr.render(ax_l, view=view, colors=((1.0, 0.647, 0.0),) * 3)

        if prev_box_moved:
            prev_box_moved.render(ax_l, view=view, colors=((0, 0, 1),) * 3)

        if scaled:
            scaled.render(ax_l, view=view, colors=((1, 0, 0),) * 3)

        # ⭐ DRAW ALIGNED BOX
        if aligned_box:
            aligned_box.render(ax_l, view=view, colors=((0, 1, 0),) * 3)

        corners = view_points(box_curr.corners(), view, normalize=False)[:2, :]

        ax_l.set_xlim(np.min(corners[0]) - margin, np.max(corners[0]) + margin)
        ax_l.set_ylim(np.min(corners[1]) - margin, np.max(corners[1]) + margin)
        ax_l.set_aspect("equal")

        # ---- BEV ----
        if prev_box and prev_box_moved:
            BEVPlotter.plot(
                ax_r,
                pts,
                prev_box,
                prev_box_moved,
                box_curr,
                scaled,
                aligned_box=result["aligned_box"],
                inside_pts=result["inside_points"],
            )

        plt.show()