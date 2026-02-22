import argparse
import time
import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from visualizer import Visualizer
from configuration import NuScenesConfig
from annotation_processor import AnnotationProcessor
from annotation_renderer import AnnotationRenderer
from box_metrics import BoxMetrics

config = NuScenesConfig()

# =====================================
#            SAMPLE MODE
# =====================================
def run_single_sample(nusc, predict_helper):
    sample = nusc.sample[150]
    ann_token = sample['anns'][13]
    ann = nusc.get('sample_annotation', ann_token)

    instance_token = ann['instance_token']
    sample_token = sample['token']

    try:
        velocity_ms = predict_helper.get_velocity_for_agent(
            instance_token,
            sample_token
        )
    except Exception:
        velocity_ms = 0.0

    velocity_kmh = velocity_ms * 3.6
    distance_m = velocity_ms * config.dt

    print(f"Speed: {velocity_kmh:.2f} km/h")
    print(f"Estimated distance moved in {config.dt*1000:.0f} ms: {distance_m:.3f} m")

    processor = AnnotationProcessor(nusc)
    result = processor.process(ann_token, distance_m)

    # ✅ compute IoU here
    iou = BoxMetrics.bev_iou(
        result["aligned_box"],
        result["box_curr"]
    )

    print(f"BEV IoU (aligned vs GT): {iou:.4f}")

    AnnotationRenderer.render(result)

# =====================================
#            DATASET MODE
# =====================================
def run_full_dataset(nusc, predict_helper):
    total_samples = len(nusc.sample)
    print(f"Total samples: {total_samples}")

    for i, sample in enumerate(nusc.sample):
        start_time = time.time()
        print(f"Processing sample {i+1}/{total_samples}", end='\r', flush=True)

        sd_token = sample['data']['LIDAR_TOP']
        ego_pose = nusc.get(
            'ego_pose',
            nusc.get('sample_data', sd_token)['ego_pose_token']
        )
        ego_translation = np.array(ego_pose['translation'])

        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)

            if not ann['category_name'].startswith('vehicle'):
                continue

            if ann['num_lidar_pts'] < config.MIN_LIDAR_POINTS:
                continue

            ann_translation = np.array(ann['translation'])
            distance = np.linalg.norm(ann_translation - ego_translation)
            if distance > config.MAX_DISTANCE:

                continue

            instance_token = ann['instance_token']
            sample_token = sample['token']

            try:
                velocity_ms = predict_helper.get_velocity_for_agent(
                    instance_token,
                    sample_token
                )
            except Exception:
                continue

            if velocity_ms is None:
                continue

            if not (config.MIN_SPEED_MS <= velocity_ms <= config.MAX_SPEED_MS):
                continue

            if ann['prev'] == '':
                continue

            prev_ann = nusc.get('sample_annotation', ann['prev'])
            if prev_ann['num_lidar_pts'] < config.MIN_LIDAR_POINTS:
                continue

            lidar_hz = 20.0
            dt = 1 / lidar_hz
            distance_m = velocity_ms * dt

            Visualizer.render_annotation(
                nusc,
                ann_token,
                distance_m,
                stepback=1
            )

        elapsed_time = time.time() - start_time

        with open(config.TIME_LOG_PATH, "a") as f:
            f.write(f"{elapsed_time:.6f}\n")


# =====================================
#                 MAIN
# =====================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["sample", "dataset"],
        default="sample",
        help="Run mode"
    )
    parser.add_argument(
        "--version",
        default="v1.0-mini",
        help="nuScenes version"
    )
    parser.add_argument(
        "--dataroot",
        default="../../nuscenes",
        help="Path to dataset"
    )

    args = parser.parse_args()

    print(f"Running mode: {args.mode}")

    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    predict_helper = PredictHelper(nusc)

    if args.mode == "sample":
        run_single_sample(nusc, predict_helper)
    else:
        run_full_dataset(nusc, predict_helper)


if __name__ == "__main__":
    main()