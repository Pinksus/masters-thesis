from dataclasses import dataclass


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
    TIME_LOG_PATH: str = "sample_processing_time.txt"

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
