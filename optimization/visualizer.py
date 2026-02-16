import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, Box
import numpy as np
import copy
from nuscenes.utils.geometry_utils import view_points
from box_utils import BoxUtils
from annotation_loader import AnnotationLoader
from box_transformer import BoxTransformer
from bev_plotter import BEVPlotter
from nuscenes.utils.geometry_utils import view_points


class Visualizer:

    @staticmethod
    def render_annotation(nusc, anntoken, distance, stepback=1, margin=10, view=np.eye(4)):

        #print("Distance is: ", distance)
        #print(type(distance))
        distance *= 10
        distance = float(distance)
        #distance = 0


        loader = AnnotationLoader(nusc)
        transformer = BoxTransformer(nusc)

        data = loader.load_current(anntoken)
        ann = data["annotation"]
        sample = data["sample"]
        box_curr = data["box"]

        prev_ann = loader.find_prev_annotation(
            ann['instance_token'],
            nusc.sample.index(sample),
            stepback
        )

        prev_box = None
        if prev_ann:
            prev_box = transformer.global_to_lidar(
                prev_ann, data["lidar_token"]
            )

        # ---- rendering ----
        pc = LidarPointCloud.from_file(data["lidar_path"])
        pts = pc.points.T[:, :3]

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 10))

        pc.render_height(ax_l, view=view)
        box_curr.render(ax_l, view=view, colors=((1.0, 0.647, 0.0),)*3)

        if prev_box:
            # --- vytvoríme kópiu prev_box a posunieme ho ---
            prev_box_moved = copy.deepcopy(prev_box)
            yaw = prev_box_moved.orientation.yaw_pitch_roll[0]  # yaw v radianoch
            direction = np.array([np.cos(yaw), np.sin(yaw)])
            prev_box_moved.center[:2] += direction * distance
            #print("Prev_box posunutý o distance:", distance, "v smere yaw", yaw)
            #print("Nové centrum prev_box:", prev_box_moved.center[:2])

            # vykreslíme posunutý prev_box na ľavom poli
            prev_box_moved.render(ax_l, view=view, colors=((0,0,1),)*3)

            # scaled box
            scaled = copy.deepcopy(prev_box_moved)
            scaled.wlh *= 1.5
            scaled.render(ax_l, view=view, colors=((1,0,0),)*3)

            ax_l.scatter(prev_box.center[0], prev_box.center[1], c='pink', s=50, label='moved center')

            ax_l.scatter(prev_box_moved.center[0], prev_box_moved.center[1], c='cyan', s=50, label='moved center')

        corners = view_points(box_curr.corners(), view, normalize=False)[:2, :]
        

        ax_l.set_xlim(np.min(corners[0]) - margin, np.max(corners[0]) + margin)
        ax_l.set_ylim(np.min(corners[1]) - margin, np.max(corners[1]) + margin)
        ax_l.set_aspect('equal')

        if prev_box:
            BEVPlotter.plot(ax_r, pts, prev_box, prev_box_moved, box_curr, scaled, distance)
        #plt.close('all') 

        plt.show()

