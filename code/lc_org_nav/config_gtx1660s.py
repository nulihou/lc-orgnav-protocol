"""
GTX 1660 Super (6GB) 优化配置

针对6GB显存显卡的优化配置，在保证性能的同时降低显存占用。
"""

from dataclasses import dataclass
from typing import List
from .config_advanced import AdvancedTrainConfig


@dataclass
class GTX1660SConfig(AdvancedTrainConfig):
    """GTX 1660 Super (6GB显存) 优化配置"""
    
    save_dir: str = "outputs/gtx1660s_checkpoints"
    model_name: str = "lc_org_gtx1660s.pt"
    exp_name: str = "gtx1660s_optimized"
    notes: str = "Optimized for GTX 1660 Super (6GB VRAM)"
    
    # ========== 显存优化：减小模型尺寸 ==========
    hidden_dim: int = 128           # 从256降到128（节省约50%显存）
    num_layers: int = 3             # 从4降到3
    num_heads: int = 4              # 从8降到4
    dropout: float = 0.1
    
    # ========== 训练优化 ==========
    batch_size: int = 1             # 保持为1（图数据通常batch=1）
    num_epochs: int = 80            # 适当减少轮数
    lr: float = 3e-4
    weight_decay: float = 1e-4
    
    # ========== 梯度累积（模拟更大batch） ==========
    grad_accum_steps: int = 2       # 累积2步，相当于batch=2
    
    # ========== 混合精度训练（必须启用！） ==========
    use_amp: bool = True            # 启用FP16，节省约40%显存
    
    # ========== EMA优化 ==========
    use_ema: bool = True
    ema_decay: float = 0.999        # 稍微降低以加快收敛
    
    # ========== 边dropout（减少计算） ==========
    use_edge_dropout: bool = True
    edge_dropout_rate: float = 0.15  # 适当增加以减少边数量
    
    # ========== 损失权重（保持不变） ==========
    lambda_goal: float = 1.0
    lambda_rel: float = 0.5
    lambda_geom: float = 0.2
    lambda_contrastive: float = 0.1   # 稍微降低
    lambda_hard_neg: float = 0.05
    
    # ========== 特性开关 ==========
    use_focal: bool = True
    use_contrastive: bool = True      # 保留，性能提升明显
    use_uncertainty: bool = True
    use_hard_neg: bool = True
    
    # ========== 学习率调度 ==========
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.01
    max_grad_norm: float = 1.0
    
    # ========== 早停（防止过拟合） ==========
    use_early_stopping: bool = True
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 0.001
    
    # ========== 评估 ==========
    top_k_list: List[int] = None
    
    def __post_init__(self):
        if self.top_k_list is None:
            self.top_k_list = [1, 3, 5]
    
    # ========== 日志 ==========
    log_interval: int = 10
    val_interval: int = 1
    save_interval: int = 5


@dataclass
class GTX1660SUltraLightConfig(GTX1660SConfig):
    """超轻量级配置 - 如果上面的配置仍然显存不足"""
    
    save_dir: str = "outputs/gtx1660s_ultra_light"
    model_name: str = "lc_org_ultra_light.pt"
    exp_name: str = "gtx1660s_ultra_light"
    notes: str = "Ultra-light config for 6GB VRAM"
    
    # 进一步减小模型
    hidden_dim: int = 96            # 更小的隐藏层
    num_layers: int = 2             # 只用2层
    num_heads: int = 4
    
    # 禁用部分特性以节省显存
    use_contrastive: bool = False   # 对比学习需要额外显存
    lambda_contrastive: float = 0.0
    
    # 更激进的边dropout
    edge_dropout_rate: float = 0.2


# 更新配置注册表
from .config_advanced import CONFIG_REGISTRY

CONFIG_REGISTRY["gtx1660s"] = GTX1660SConfig
CONFIG_REGISTRY["gtx1660s_ultra_light"] = GTX1660SUltraLightConfig

