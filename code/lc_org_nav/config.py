
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Paths
    train_list: str = "data/train_list.txt"   # each line: path to a .npz sample
    val_list: str = "data/val_list.txt"
    save_dir: str = "outputs/checkpoints"
    model_name: str = "lc_org_goal_selector.pt"

    # Model dims
    node_feat_dim: int = 32   # must match data
    edge_feat_dim: int = 8
    lang_feat_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 3
    num_rel_types: int = 5    # e.g. near/front/behind/left/right

    # Training
    num_epochs: int = 80
    batch_size: int = 1       # keep 1 for simplicity; each batch is one scene
    lr: float = 1e-4
    weight_decay: float = 1e-5
    device: str = "cuda"      # or "cpu"

    # Loss weights
    lambda_rel: float = 0.5
    lambda_geom: float = 0.1
    lambda_rank: float = 0.0  # Ranking loss weight (for multi-candidate optimization)
    rank_margin: float = 1.0  # Margin for ranking loss
    
    # 训练阶段是否使用关系一致掩码（重要：默认 False，确保使用干净的 candidate_mask）
    # 模式 A（默认推荐）：use_relation_mask=False
    #   - 训练：不使用"关系一致候选掩码"，在原始 candidate_mask 下从完整正负样本学习
    #   - 验证/测试：只用 .npz 的原始 candidate_mask，绝不二次收紧
    # 模式 B（仅用于诊断/对比）：use_relation_mask=True
    #   - 训练：使用"关系一致候选掩码"收紧候选，加入规则先验辅助收敛
    #   - 验证/测试：仍然只用原始 candidate_mask（避免规则泄露）
    use_relation_mask: bool = False
    
    # Logging
    log_interval: int = 20
    val_interval: int = 1
    
    # Model architecture options
    use_rel_gate: bool = False  # 是否使用关系 Gate 模块（轻量级，对 1660S 友好）


    use_edge_context_head: bool = False  # Explicit raw-edge-context residual scoring head

@dataclass
class GoalOnlyConfig(TrainConfig):
    """专用于 GoalOnly 训练的独立配置（仅训练目标选择头）"""
    # 覆盖保存目录与损失配置
    save_dir: str = "outputs/goal_only"
    lambda_rel: float = 0.0
    lambda_geom: float = 0.0
    # 训练阶段是否使用关系一致掩码（重要：本版设为 False，训练不依赖掩码）
    use_relation_mask: bool = False

    # Logging
    log_interval: int = 20
    val_interval: int = 1
