class AnnotationLoader:

    def __init__(self, nusc):
        self.nusc = nusc

    def load_current(self, anntoken):
        ann = self.nusc.get('sample_annotation', anntoken)
        sample = self.nusc.get('sample', ann['sample_token'])

        lidar_token = sample['data']['LIDAR_TOP']
        lidar_path, boxes, _ = self.nusc.get_sample_data(
            lidar_token,
            selected_anntokens=[anntoken]
        )

        return {
            "annotation": ann,
            "sample": sample,
            "lidar_token": lidar_token,
            "lidar_path": lidar_path,
            "box": boxes[0]
        }

    def find_prev_annotation(self, instance_token, sample_idx, stepback):
        target_idx = sample_idx - stepback
        if target_idx < 0:
            return None

        prev_sample = self.nusc.sample[target_idx]
        for ann_token in prev_sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            if ann['instance_token'] == instance_token:
                return ann

        return None
