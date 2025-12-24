from nuscenes.nuscenes import NuScenes
from nuscenes_utils import NuScenesUtils
from visualizer import Visualizer


# =====================================
#            EXAMPLE USAGE
# =====================================
nusc = NuScenes(version='v1.0-mini', dataroot='../../nuscenes')
nusc_utils = NuScenesUtils(nusc)

# vyber sample a anotáciu
sample = nusc.sample[56]
ann_token = sample['anns'][12]

Visualizer.render_annotation(nusc, ann_token, stepback=1)
