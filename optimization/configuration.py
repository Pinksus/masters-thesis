from dataclasses import dataclass
from datetime import datetime


@dataclass
class NuScenesConfig:

    # ================================
    # Dataset
    # ================================
    NUSC_VERSION: str = "v1.0-trainval"
    NUSC_PATH: str = "../../Nuscene"

    # ================================
    # Object filtering
    # ================================
    MAX_DISTANCE: float = 50.0
    MIN_LIDAR_POINTS: int = 5

    # ================================
    # Velocity filtering
    # ================================
    MIN_SPEED_KMH: float = 5.0
    MAX_SPEED_KMH: float = 100.0
    MIN_SPEED_MS: float = MIN_SPEED_KMH / 3.6
    MAX_SPEED_MS: float = MAX_SPEED_KMH / 3.6

    # ================================
    # LiDAR timing
    # ================================
    LIDAR_HZ: float = 20.0

    # ================================
    # Rendering
    # ================================
    STEPBACK: int = 1

    # ================================
    # Logging
    # ================================
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M")

    TIME_LOG_PATH: str = f"outputs/{timestamp}_sample_processing_time.txt"
    IOU_LOG_PATH: str = f"outputs/{timestamp}_iou_log.txt"

    # ================================
    # Debug
    # ================================
    PRINT_PROCESS: bool = True

    # ================================
    # Derived values (auto computed)
    # ================================
    @property
    def dt(self) -> float:
        return 1.0 / self.LIDAR_HZ

    @property
    def min_speed_ms(self) -> float:
        return self.MIN_SPEED_KMH / 3.6

    @property
    def max_speed_ms(self) -> float:
        return self.MAX_SPEED_KMH / 3.6
