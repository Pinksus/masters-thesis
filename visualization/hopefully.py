import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import numpy as np
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.color_map import get_colormap
from PIL import Image
import copy




def get_color(category_name: str) -> tuple[int, int, int]:
        """
        Provides the default colors based on the category names.
        This method works for the general nuScenes categories, as well as the nuScenes detection categories.
        """

        return nusc.colormap[category_name]


def render_annotation(nusc,
                             anntoken: str,
                             margin: float = 10,
                             view: np.ndarray = np.eye(4),
                             box_vis_level: BoxVisibility = BoxVisibility.ANY):
    """
    Render selected annotation using a *single* plot (LIDAR view only).
    """
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'], 'Error: No LIDAR_TOP in data.'

    # Find camera where annotation is visible (needed for nuScenes API constraints)
    cams = [key for key in sample_record['data'] if 'CAM' in key]
    found = False
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(
            sample_record['data'][cam],
            box_vis_level=box_vis_level,
            selected_anntokens=[anntoken]
        )
        if len(boxes) == 1:
            found = True
            break

    assert found, "Annotation not visible in any camera image. Try BoxVisibility.ANY."

    # Prepare figure (single axis)
    fig, ax = plt.subplots(figsize=(12, 12))

    # LIDAR TOP data
    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, _ = nusc.get_sample_data(
        lidar,
        selected_anntokens=[anntoken]
    )

    # Render height map
    LidarPointCloud.from_file(data_path).render_height(ax, view=view)

    prev_boxes = []
    # Draw annotation box
    for box in boxes:
        # Original box (draw)
        c = np.array(get_color(box.name)) / 255.0
        box.render(ax, view=view, colors=(c, c, c))

        # Duplicate box
        new_box = copy.deepcopy(box)

        # Modify x,y,z (xyz)
        new_box.center[0] -= 0.8  # example: shift x by +0.5
        new_box.center[1] -= 0.8 # shift y if needed
        #ew_box.center[2] += ... # shift z if needed

        # rotate by +10 degrees yaw
        yaw = np.deg2rad(-13)
        q_rot = Quaternion(axis=[0, 0, 1], angle=yaw)

        # apply rotation
        new_box.orientation = q_rot * new_box.orientation

        prev_boxes.append(new_box)
        # Choose a different color
        c2 = np.array([255, 255, 0]) / 255.0  # yellow

        # Draw duplicated box
        new_box.render(ax, view=view, colors=(c2, c2, c2))


    print(f"Current box:\n{boxes}")
    print(f"Previous box:\n{prev_boxes}")
    # Autoscale around box
    corners = view_points(boxes[0].corners(), view, False)[:2, :]
    ax.set_xlim([np.min(corners[0]) - margin, np.max(corners[0]) + margin])
    ax.set_ylim([np.min(corners[1]) - margin, np.max(corners[1]) + margin])

    ax.axis('off')
    ax.set_aspect('equal')

    plt.show()



def render_lidar_all_annotations(
        nusc,
        anntoken,
        prev_ann_token=None,
        margin=10,
        view: np.ndarray = np.eye(4),
        box_vis_level: BoxVisibility = BoxVisibility.ANY
    ):
    """
    Render full LIDAR_TOP point cloud with ALL annotation boxes.
    Full BEV view (not zoomed on selected box).
    """

    # --- Get sample containing this annotation ---
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data']

    # --- Required check: annotation must be visible in a camera ---
    cams = [key for key in sample_record['data'] if 'CAM' in key]
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(
            sample_record['data'][cam],
            box_vis_level=box_vis_level,
            selected_anntokens=[anntoken]
        )
        if len(boxes) == 1:
            break

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(12, 12))
    print(boxes)

    # --- Load full LIDAR point cloud and all boxes ---
    lidar_token = sample_record['data']['LIDAR_TOP']
    data_path, boxes_all, _ = nusc.get_sample_data(lidar_token)

    # --- Render LIDAR height map ---
    LidarPointCloud.from_file(data_path).render_height(ax, view=view)

    print(f"Total boxes in scene: {len(boxes_all)}")

    # --- Render every annotation box ---
    for box in boxes_all:
        c = np.array(get_color(box.name)) / 255.0

        # Highlight current annotation (red)
        if box.token == anntoken:
            c = np.array([1.0, 0.0, 0.0])

        # Highlight previous annotation (blue)
        if prev_ann_token is not None and box.token == prev_ann_token:
            c = np.array([0.0, 0.0, 1.0])

        box.render(ax, view=view, colors=(c, c, c))

    # --- Compute full BEV extents from all boxes ---
    all_pts = []
    for box in boxes_all:
        pts = view_points(box.corners(), view, False)[:2, :]
        all_pts.append(pts)

    all_pts = np.concatenate(all_pts, axis=1)

    ax.set_xlim([np.min(all_pts[0]) - margin, np.max(all_pts[0]) + margin])
    ax.set_ylim([np.min(all_pts[1]) - margin, np.max(all_pts[1]) + margin])

    # Or: fixed size BEV window
    # ax.set_xlim([-60, 60])
    # ax.set_ylim([-60, 60])

    ax.axis('off')
    ax.set_aspect('equal')

    plt.show()


def get_nth_previous_ann(nusc, ann_token, n=3):
    """
    Repeat 'prev' pointer lookup n times to get earlier annotations.
    """
    token = ann_token
    for _ in range(n):
        record = nusc.get('sample_annotation', token)
        if record['prev'] == "":
            return None  # no earlier frame
        token = record['prev']
    return token


# =========================================================
#                  RUN THE SCRIPT
# =========================================================
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='../nuscenes')

# ---------- SAMPLE 56 ----------
sample = nusc.sample[56]
ann_token = sample['anns'][12]

prev_ann_token = get_nth_previous_ann(nusc, ann_token)

print("Sample 56 annotation:", ann_token)
print("3rd previous annotation:", prev_ann_token)

#render_lidar_all_annotations(nusc, ann_token, prev_ann_token)
render_annotation(nusc,ann_token)

'''# ---------- SAMPLE 53 ----------
sample = nusc.sample[53]
ann_token = sample['anns'][12]

prev_ann_token = get_nth_previous_ann(nusc, ann_token)

print("Sample 53 annotation:", ann_token)
print("3rd previous annotation:", prev_ann_token)

render_lidar_all_annotations(nusc, ann_token, prev_ann_token)
'''
plt.show()

