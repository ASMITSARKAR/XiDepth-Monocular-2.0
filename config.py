from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    name: str = "xidepth"  # "xidepth" or "resnet18"
    num_scales: int = 4
    min_depth: float = 0.1
    max_depth: float = 100.0
    disp_bias_init: float = -4.5
    pretrained_encoder: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 12
    num_epochs: int = 20
    learning_rate: float = 1e-4
    lr_step_size: int = 15
    lr_gamma: float = 0.1
    weight_decay: float = 1e-4
    use_amp: bool = True
    num_workers: int = 4

    # Loss terms
    ssim_weight: float = 0.85
    l1_weight: float = 0.15
    smoothness_weight: float = 0.001
    use_automask: bool = True

    # Health monitor thresholds
    warmup_steps: int = 1000
    dead_neuron_std_thresh: float = 1e-6
    max_consecutive_collapse: int = 100
    fail_on_collapse: bool = False


@dataclass
class DatasetConfig:
    height: int = 192
    width: int = 640
    frame_ids: List[int] = field(default_factory=lambda: [0, -1, 1])
    use_stereo: bool = True
    split_dir: str = "data/splits/eigen_zhou"
