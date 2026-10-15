"""
Advanced configuration for LC-ORGNav with comprehensive options.

Supports multiple experimental configurations for ablation studies.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AdvancedTrainConfig:
    """Advanced training configuration with all bells and whistles."""
    
    # ========== Paths ==========
    train_list: str = "data/train_list.txt"
    val_list: str = "data/val_list.txt"
    save_dir: str = "outputs/advanced_checkpoints"
    model_name: str = "lc_org_advanced.pt"
    
    # ========== Model Architecture ==========
    node_feat_dim: int = 32  # Auto-inferred from data
    edge_feat_dim: int = 8   # Auto-inferred from data
    lang_feat_dim: int = 32  # Auto-inferred from data
    hidden_dim: int = 256    # Increased for better capacity
    num_layers: int = 4      # Deeper network
    num_rel_types: int = 5
    num_heads: int = 8       # Multi-head attention
    dropout: float = 0.15
    use_edge_dropout: bool = True
    edge_dropout_rate: float = 0.1
    
    # ========== Training Hyperparameters ==========
    num_epochs: int = 100
    batch_size: int = 1      # Keep 1 for graph data
    lr: float = 3e-4         # Slightly higher for larger model
    weight_decay: float = 1e-4
    device: str = "cuda"
    
    # ========== Loss Weights ==========
    lambda_goal: float = 1.0
    lambda_rel: float = 0.5
    lambda_geom: float = 0.2
    lambda_contrastive: float = 0.15
    lambda_hard_neg: float = 0.1
    
    # ========== Loss Configuration ==========
    use_focal: bool = True
    use_contrastive: bool = True
    use_uncertainty: bool = True
    use_hard_neg: bool = True
    
    # ========== Optimizer & Scheduler ==========
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.01
    
    # ========== Advanced Training Strategies ==========
    use_amp: bool = True              # Mixed precision training
    use_ema: bool = True              # Exponential moving average
    ema_decay: float = 0.9995
    max_grad_norm: float = 1.0        # Gradient clipping
    grad_accum_steps: int = 1         # Gradient accumulation
    
    # ========== Early Stopping ==========
    use_early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    
    # ========== Evaluation ==========
    top_k_list: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    
    # ========== Logging ==========
    log_interval: int = 20
    val_interval: int = 1
    save_interval: int = 5
    
    # ========== Experiment Name ==========
    exp_name: str = "advanced_full"
    notes: str = "Full advanced model with all features"


@dataclass
class BaselineConfig(AdvancedTrainConfig):
    """Baseline configuration for ablation studies."""
    
    save_dir: str = "outputs/baseline_checkpoints"
    model_name: str = "lc_org_baseline.pt"
    exp_name: str = "baseline"
    notes: str = "Baseline model without advanced features"
    
    # Simplified architecture
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    
    # No advanced features
    use_focal: bool = False
    use_contrastive: bool = False
    use_uncertainty: bool = False
    use_hard_neg: bool = False
    use_ema: bool = False
    use_edge_dropout: bool = False
    
    # Simpler loss
    lambda_contrastive: float = 0.0
    lambda_hard_neg: float = 0.0


@dataclass
class AblationNoAttentionConfig(AdvancedTrainConfig):
    """Ablation: Remove cross-attention mechanism."""
    
    save_dir: str = "outputs/ablation_no_attention"
    model_name: str = "lc_org_no_attention.pt"
    exp_name: str = "ablation_no_attention"
    notes: str = "Ablation study: no cross-attention"
    
    num_heads: int = 1  # Effectively disables multi-head attention


@dataclass
class AblationNoContrastiveConfig(AdvancedTrainConfig):
    """Ablation: Remove contrastive learning."""
    
    save_dir: str = "outputs/ablation_no_contrastive"
    model_name: str = "lc_org_no_contrastive.pt"
    exp_name: str = "ablation_no_contrastive"
    notes: str = "Ablation study: no contrastive loss"
    
    use_contrastive: bool = False
    lambda_contrastive: float = 0.0


@dataclass
class AblationNoUncertaintyConfig(AdvancedTrainConfig):
    """Ablation: Remove uncertainty estimation."""
    
    save_dir: str = "outputs/ablation_no_uncertainty"
    model_name: str = "lc_org_no_uncertainty.pt"
    exp_name: str = "ablation_no_uncertainty"
    notes: str = "Ablation study: no uncertainty estimation"
    
    use_uncertainty: bool = False


@dataclass
class AblationSimplePoolingConfig(AdvancedTrainConfig):
    """Ablation: Use simple pooling instead of hierarchical."""
    
    save_dir: str = "outputs/ablation_simple_pooling"
    model_name: str = "lc_org_simple_pooling.pt"
    exp_name: str = "ablation_simple_pooling"
    notes: str = "Ablation study: simple max pooling only"


@dataclass
class SmallModelConfig(AdvancedTrainConfig):
    """Small model for faster experimentation."""
    
    save_dir: str = "outputs/small_model"
    model_name: str = "lc_org_small.pt"
    exp_name: str = "small_model"
    notes: str = "Small model for fast iteration"
    
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    num_epochs: int = 50


@dataclass
class LargeModelConfig(AdvancedTrainConfig):
    """Large model for maximum performance."""
    
    save_dir: str = "outputs/large_model"
    model_name: str = "lc_org_large.pt"
    exp_name: str = "large_model"
    notes: str = "Large model for best performance"
    
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 16
    dropout: float = 0.2
    num_epochs: int = 150
    lr: float = 1e-4


@dataclass
class OverfitOneConfig(AdvancedTrainConfig):
    """Single-sample overfit config to sanity-check loss/labels/gradients."""
    
    save_dir: str = "outputs/overfit_one"
    model_name: str = "lc_org_overfit1.pt"
    exp_name: str = "overfit_one_sample"
    notes: str = "Train on 1 sample, batch=1, long epochs; aux losses off"

    # Data: single-sample lists (will be created before running)
    train_list: str = "data/seen_overfit_train_list.txt"
    val_list: str = "data/seen_overfit_train_list.txt"

    # Train small, focused
    num_epochs: int = 200
    batch_size: int = 1
    lr: float = 3e-4

    # Disable auxiliary losses to focus on goal selection fit
    lambda_goal: float = 1.0
    lambda_rel: float = 0.0
    lambda_geom: float = 0.0
    lambda_contrastive: float = 0.0
    lambda_hard_neg: float = 0.0

    use_focal: bool = False
    use_contrastive: bool = False
    use_uncertainty: bool = False
    use_hard_neg: bool = False

    # Simplify training dynamics
    use_ema: bool = False
    grad_accum_steps: int = 1
    max_grad_norm: float = 0.0

    # Frequent validation to observe fit
    log_interval: int = 1
    val_interval: int = 1
    save_interval: int = 1

    # Disable export and visual reports for speed
    export_on_best: bool = False
    report_milestones: List[int] = field(default_factory=list)

    # Disable early stopping to force full 200 epochs
    use_early_stopping: bool = False

@dataclass
class GoalOnlyConfig(AdvancedTrainConfig):
    """Train only the goal selection head (disable auxiliary tasks)."""
    
    save_dir: str = "outputs/goal_only"
    model_name: str = "lc_org_goal_only.pt"
    exp_name: str = "goal_only"
    notes: str = "Only goal selection loss; auxiliaries disabled"

    # Use seen train/val lists
    train_list: str = "data/seen_train_list.txt"
    val_list: str = "data/seen_val_list.txt"

    # Loss weights: keep only goal
    lambda_goal: float = 1.0
    lambda_rel: float = 0.0
    lambda_geom: float = 0.0
    lambda_contrastive: float = 0.0
    lambda_hard_neg: float = 0.0

    # Disable auxiliary loss modules
    use_focal: bool = False
    use_contrastive: bool = False
    use_uncertainty: bool = False
    use_hard_neg: bool = False

    # Save and validate frequently
    val_interval: int = 1
    save_interval: int = 5

    # Disable early stopping to observe full training
    use_early_stopping: bool = False
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0005

    # Disable EMA for simplicity
    use_ema: bool = False

    # Disable export and milestone reports to avoid extra overhead
    export_on_best: bool = False
    report_milestones: List[int] = field(default_factory=list)

# Configuration registry for easy access
CONFIG_REGISTRY = {
    "advanced": AdvancedTrainConfig,
    "baseline": BaselineConfig,
    "ablation_no_attention": AblationNoAttentionConfig,
    "ablation_no_contrastive": AblationNoContrastiveConfig,
    "ablation_no_uncertainty": AblationNoUncertaintyConfig,
    "ablation_simple_pooling": AblationSimplePoolingConfig,
    "small": SmallModelConfig,
    "large": LargeModelConfig,
    "overfit1": OverfitOneConfig,
    "goal_only": GoalOnlyConfig,
    # GTX 1660S configs will be added by config_gtx1660s.py
}


def get_config(name: str = "advanced") -> AdvancedTrainConfig:
    """Get configuration by name.
    
    Args:
        name: Configuration name from CONFIG_REGISTRY
        
    Returns:
        Configuration instance
    """
    if name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIG_REGISTRY.keys())}")
    
    return CONFIG_REGISTRY[name]()

