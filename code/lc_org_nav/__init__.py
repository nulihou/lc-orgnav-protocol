"""
LC-ORGNav: Language-Conditioned Object-Relation Graph Navigation

A state-of-the-art system for language-conditioned goal selection in
autonomous driving scenarios using advanced graph neural networks.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Basic imports
from .config import TrainConfig
from .dataset import GoalSelectionDataset
from .model import LCORGNet
try:
    from .trainer import GoalSelectorTrainer
except ImportError:
    GoalSelectorTrainer = None

# Advanced imports
try:
    from .config_advanced import (
        AdvancedTrainConfig,
        BaselineConfig,
        get_config,
        CONFIG_REGISTRY,
    )
except ImportError:
    AdvancedTrainConfig = None
    BaselineConfig = None
    get_config = None
    CONFIG_REGISTRY = None

# GTX 1660S optimized configs
try:
    from .config_gtx1660s import GTX1660SConfig, GTX1660SUltraLightConfig
except ImportError:
    pass  # Optional module
try:
    from .model_advanced import LCORGNetAdvanced
except ImportError:
    LCORGNetAdvanced = None
try:
    from .trainer_advanced import AdvancedTrainer
except ImportError:
    AdvancedTrainer = None
try:
    from .losses import CombinedLoss
except ImportError:
    CombinedLoss = None
try:
    from .metrics import MetricsCalculator
except ImportError:
    MetricsCalculator = None
try:
    from .inference import InferenceEngine, load_model_for_inference
except ImportError:
    InferenceEngine = None
    load_model_for_inference = None
try:
    from .visualization import TrainingVisualizer, SceneGraphVisualizer
except ImportError:
    TrainingVisualizer = None
    SceneGraphVisualizer = None
try:
    from .data_augmentation import (
        GraphAugmentation,
        HardNegativeSampler,
        SyntheticSceneGenerator,
    )
except ImportError:
    GraphAugmentation = None
    HardNegativeSampler = None
    SyntheticSceneGenerator = None

__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Basic
    "TrainConfig",
    "GoalSelectionDataset",
    "LCORGNet",
    "GoalSelectorTrainer",
    
    # Advanced
    "AdvancedTrainConfig",
    "BaselineConfig",
    "get_config",
    "CONFIG_REGISTRY",
    "LCORGNetAdvanced",
    "AdvancedTrainer",
    "CombinedLoss",
    "MetricsCalculator",
    "InferenceEngine",
    "load_model_for_inference",
    "TrainingVisualizer",
    "SceneGraphVisualizer",
    "GraphAugmentation",
    "HardNegativeSampler",
    "SyntheticSceneGenerator",
]
