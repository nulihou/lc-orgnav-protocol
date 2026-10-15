
import os
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import numpy as np

from .config import TrainConfig
from .dataset import GoalSelectionDataset
from .model import LCORGNet
import math
import torch

def _move_to_device(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            out[k] = [x.to(device) for x in v]
        else:
            out[k] = v
    return out

class GoalSelectorTrainer:
    """
    正式 Baseline Trainer（区分两种模式，评估阶段严禁使用关系掩码）
    - 模式 A（默认推荐）：cfg.use_relation_mask=False
      训练：不使用“关系一致候选掩码”，在原始 candidate_mask 下从完整正负样本学习
      验证/测试：只用 .npz 的原始 candidate_mask，绝不二次收紧
    - 模式 B（可做对比/诊断）：cfg.use_relation_mask=True
      训练：使用“关系一致候选掩码”收紧候选，加入规则先验辅助收敛
      验证/测试：仍然只用原始 candidate_mask（避免规则泄露）
    """
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        self.device = device
        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0
        self.last_val_gs3: float = 0.0

        # datasets
        self.train_ds = GoalSelectionDataset(cfg.train_list)
        self.val_ds = GoalSelectionDataset(cfg.val_list)

        # Prepare sampler if configured
        sampler = None
        shuffle = True
        
        if getattr(cfg, 'use_weighted_sampler', False):
            print("[Trainer] Initializing WeightedRandomSampler...")
            weights = []
            high_k_count = 0
            high_k_thresh = getattr(cfg, 'high_k_threshold', 4)
            high_k_w = getattr(cfg, 'high_k_weight', 10.0)
            
            print(f"[Trainer] Scanning {len(self.train_ds.paths)} files for K>={high_k_thresh}...")
            # Scan files to calculate weights
            # This might take a moment but is necessary for correct sampling
            for i, p in enumerate(self.train_ds.paths):
                if i % 1000 == 0:
                    print(f"  scanned {i} files...")
                try:
                    data = np.load(p, allow_pickle=True)
                    k = 0
                    if 'candidate_mask' in data:
                        # Handle potential extra batch dim if present (though unlikely in raw npz unless saved that way)
                        cm = data['candidate_mask']
                        if cm.ndim == 2 and cm.shape[0] == 1:
                            cm = cm[0]
                        k = int(np.sum(cm))
                    elif 'num_candidates' in data:
                        k = int(data['num_candidates'])
                    
                    if k >= high_k_thresh:
                        weights.append(high_k_w)
                        high_k_count += 1
                    else:
                        weights.append(1.0)
                except Exception as e:
                    # Fallback to weight 1.0 if error
                    weights.append(1.0)
            
            print(f"[Trainer] WeightedRandomSampler ready. High-K (K>={high_k_thresh}) samples: {high_k_count}/{len(weights)}")
            weights_tensor = torch.DoubleTensor(weights)
            sampler = WeightedRandomSampler(weights_tensor, len(weights_tensor), replacement=True)
            shuffle = False

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # infer feature dimensions from first sample
        sample0 = self.train_ds[0]
        node_feat_dim = sample0["node_feats"].shape[1]
        edge_feat_dim = sample0["edge_attr"].shape[1]
        lang_feat_dim = sample0["lang_feat"].shape[0]

        self.model = LCORGNet(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            lang_feat_dim=lang_feat_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_rel_types=cfg.num_rel_types,
            use_edge_context_head=getattr(cfg, 'use_edge_context_head', False),
            use_rel_gate=getattr(cfg, 'use_rel_gate', False),  # 支持关系 Gate
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        os.makedirs(cfg.save_dir, exist_ok=True)

    @staticmethod
    def _build_relation_mask_from_anchor_feats(node_feats: torch.Tensor, rel_type: int) -> torch.Tensor:
        """
        使用 anchor-relative 特征重建关系一致的候选掩码。
        期望列布局：7:d_a, 8:cos_a, 9:sin_a, 11:is_anchor
        """
        d_a = node_feats[:, 7]
        cos_a = node_feats[:, 8]
        sin_a = node_feats[:, 9]
        is_anchor_col = node_feats[:, 11] if node_feats.size(1) > 11 else torch.zeros_like(d_a)
        theta = torch.atan2(sin_a, cos_a)
        near_d = 20.0
        far_min_d = 5.0
        far_max_d = 40.0
        front_deg = math.radians(30.0)
        side_min = math.radians(60.0)
        side_max = math.radians(120.0)
        if rel_type == 0:
            ok = d_a < near_d
        elif rel_type == 1:
            ok = (d_a > far_min_d) & (d_a < far_max_d) & (theta.abs() < front_deg)
        elif rel_type == 2:
            ok = (d_a > far_min_d) & (d_a < far_max_d) & ((theta.abs() - math.pi).abs() < front_deg)
        elif rel_type == 3:
            ok = (d_a > far_min_d) & (d_a < far_max_d) & ((theta > side_min) & (theta < side_max))
        elif rel_type == 4:
            ok = (d_a > far_min_d) & (d_a < far_max_d) & ((theta > -side_max) & (theta < -side_min))
        else:
            ok = torch.zeros_like(d_a, dtype=torch.bool)
        # 过滤掉 anchor 本身
        ok = ok & (is_anchor_col < 0.5)
        return ok

    def _maybe_override_candidate_mask(self, sample: dict) -> dict:
        """若配置要求，使用关系一致的掩码替换/收紧 candidate_mask（同时保证 target 在候选内）。"""
        use_mask = getattr(self.cfg, "use_relation_mask", False)
        # 调试信息：只在第一次调用时打印一次
        if not hasattr(self, "_mask_debug_printed"):
            print(f"[DEBUG] use_relation_mask = {use_mask}")
            self._mask_debug_printed = True
        if not use_mask:
            return sample
        node_feats = sample["node_feats"]
        # 兼容 batch 维度（batch_size=1），在计算时去掉batch维
        had_batch = False
        if node_feats.dim() == 3 and node_feats.size(0) == 1:
            had_batch = True
            node_feats_2d = node_feats[0]
        else:
            node_feats_2d = node_feats
        rel_type = int(sample["rel_type"]) if not isinstance(sample["rel_type"], torch.Tensor) else int(sample["rel_type"].item())
        mask_rel = self._build_relation_mask_from_anchor_feats(node_feats_2d, rel_type)
        # 交集：在原候选集合基础上收紧为关系一致集合，但确保 target 一定为 True
        cand = sample["candidate_mask"].clone()
        if cand.dim() == 2 and cand.size(0) == 1:
            cand_1d = cand[0]
        else:
            cand_1d = cand
        cand_1d = cand_1d & mask_rel
        target_index = int(sample["target_index"]) if not isinstance(sample["target_index"], torch.Tensor) else int(sample["target_index"].item())
        if 0 <= target_index < cand_1d.numel():
            cand_1d[target_index] = True
        # 写回形状
        sample["candidate_mask"] = cand_1d.unsqueeze(0) if had_batch else cand_1d
        return sample

    def compute_losses(
        self,
        outputs: Dict[str, Any],
        sample: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Compute goal, relation, and geometric losses for a batch.

        We assume batch_size = 1 for simplicity; if larger, we iterate over batch.
        """
        goal_logits = outputs["goal_logits"]
        rel_type_logits = outputs["rel_type_logits"]
        geom_pred = outputs["geom_pred"]

        if isinstance(goal_logits, torch.Tensor) and goal_logits.dim() == 1:
            # (N,), (), () cases: wrap into batch dim of size 1
            goal_logits = goal_logits.unsqueeze(0)
            rel_type_logits = rel_type_logits.unsqueeze(0)
            geom_pred = geom_pred.unsqueeze(0)

        # Handle list of logits (from variable-sized batch) or stacked tensor
        if isinstance(goal_logits, list):
            B = len(goal_logits)
        else:
            B = goal_logits.size(0)
            
        losses = {
            "goal": torch.tensor(0.0, device=self.device),
            "rel": torch.tensor(0.0, device=self.device),
            "geom": torch.tensor(0.0, device=self.device),
            "rank": torch.tensor(0.0, device=self.device),
        }

        total_samples = 0
        for b in range(B):
            logits_nodes = goal_logits[b]          # (N,)
            rel_logits_b = rel_type_logits[b]      # (R,)
            geom_pred_b = geom_pred[b]             # (2,)

            # Handle both list and tensor formats for candidate_mask
            if isinstance(sample["candidate_mask"], list):
                candidate_mask = sample["candidate_mask"][b]
            elif isinstance(sample["candidate_mask"], torch.Tensor) and sample["candidate_mask"].dim() == 2:
                candidate_mask = sample["candidate_mask"][b]
            else:
                 # Singleton batch case where dim=1 (N,)
                candidate_mask = sample["candidate_mask"]
            
            # Similar handling for target_index
            if isinstance(sample["target_index"], list):
                target_index = sample["target_index"][b]
            elif isinstance(sample["target_index"], torch.Tensor) and sample["target_index"].dim() > 0:
                target_index = sample["target_index"][b].item()
            else:
                target_index = sample["target_index"]
                
            # rel_type
            if isinstance(sample["rel_type"], list):
                rel_type = sample["rel_type"][b]
            elif isinstance(sample["rel_type"], torch.Tensor) and sample["rel_type"].dim() > 0:
                rel_type = sample["rel_type"][b].item()
            else:
                rel_type = sample["rel_type"]
                
            # geom
            if isinstance(sample["geom"], list):
                geom_gt = sample["geom"][b]
            elif isinstance(sample["geom"], torch.Tensor) and sample["geom"].dim() == 2:
                geom_gt = sample["geom"][b]
            else:
                geom_gt = sample["geom"]

            # goal loss: restrict logits to candidate nodes
            # Note: logits_nodes might contain -inf padding if we used padded stacking,
            # but here we retrieve from list or padded tensor. 
            # If padded tensor, we need to mask out padding, but we rely on candidate_mask
            
            cand_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
            if cand_indices.numel() == 0:
                # skip if no candidates; this sample is invalid
                continue
            
            # Ensure logits_nodes is valid (if padded, we should only index valid parts)
            # cand_indices should be within valid range of N
            logits_cand = logits_nodes[cand_indices]   # (Nc,)

            # map global target_index to local index
            # find where cand_indices == target_index
            matches = torch.nonzero(cand_indices == int(target_index), as_tuple=False).squeeze(-1)
            if matches.numel() == 0:
                # target is not in candidates; skip
                continue
            target_local_idx = matches[0].item()
            target_local_tensor = torch.tensor([target_local_idx], device=self.device, dtype=torch.long)

            loss_goal_b = F.cross_entropy(
                logits_cand.unsqueeze(0),  # (1, Nc)
                target_local_tensor,
            )

            # relation type loss
            rel_type_tensor = torch.tensor([int(rel_type)], device=self.device, dtype=torch.long)
            loss_rel_b = F.cross_entropy(
                rel_logits_b.unsqueeze(0),  # (1, R)
                rel_type_tensor,
            )

            # geometric regression loss
            loss_geom_b = F.smooth_l1_loss(geom_pred_b, geom_gt.to(self.device))

            # ranking loss: force target logit to be higher than negative candidates by margin
            loss_rank_b = torch.tensor(0.0, device=self.device)
            if hasattr(self.cfg, 'lambda_rank') and self.cfg.lambda_rank > 0:
                target_logit = logits_cand[target_local_idx]  # scalar
                # Get negative candidate logits (all except target)
                neg_mask = torch.ones(len(logits_cand), dtype=torch.bool, device=self.device)
                neg_mask[target_local_idx] = False
                neg_logits = logits_cand[neg_mask]
                
                if neg_logits.numel() > 0:
                    margin = getattr(self.cfg, 'rank_margin', 1.0)
                    # L_rank = sum_i max(0, margin - (s_target - s_neg_i))
                    rank_diff = margin - (target_logit.unsqueeze(0) - neg_logits)
                    loss_rank_b = torch.clamp(rank_diff, min=0.0).mean()

            losses["goal"] += loss_goal_b
            losses["rel"] += loss_rel_b
            losses["geom"] += loss_geom_b
            losses["rank"] += loss_rank_b
            total_samples += 1

        if total_samples > 0:
            for k in losses:
                losses[k] = losses[k] / total_samples

        return losses

    def train(self):
        cfg = self.cfg
        for epoch in range(1, cfg.num_epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_goal = 0.0
            total_rel = 0.0
            total_geom = 0.0
            total_rank = 0.0
            num_batches = 0

            for batch_idx, sample in enumerate(self.train_loader, start=1):
                sample = _move_to_device(sample, self.device)
                # 训练阶段掩码策略：
                # - 模式 A（cfg.use_relation_mask=False）：此调用为空操作（不收紧候选）
                # - 模式 B（cfg.use_relation_mask=True）：按几何规则收紧候选（仅训练阶段）
                sample = self._maybe_override_candidate_mask(sample)

                outputs = self.model(
                    node_feats=sample["node_feats"],
                    edge_index=sample["edge_index"],
                    edge_attr=sample["edge_attr"],
                    lang_feat=sample["lang_feat"],
                    candidate_mask=sample["candidate_mask"],
                )

                losses = self.compute_losses(outputs, sample)
                loss = (
                    losses["goal"]
                    + cfg.lambda_rel * losses["rel"]
                    + cfg.lambda_geom * losses["geom"]
                    + getattr(cfg, 'lambda_rank', 0.0) * losses["rank"]
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_goal += losses["goal"].item()
                total_rel += losses["rel"].item()
                total_geom += losses["geom"].item()
                total_rank += losses["rank"].item()
                num_batches += 1

                if batch_idx % cfg.log_interval == 0:
                    rank_str = f" | Rank: {losses['rank'].item():.4f}" if getattr(cfg, 'lambda_rank', 0.0) > 0 else ""
                    print(
                        f"Epoch {epoch:03d} | Batch {batch_idx:04d} | "
                        f"Loss: {loss.item():.4f} | Goal: {losses['goal'].item():.4f} | "
                        f"Rel: {losses['rel'].item():.4f} | Geom: {losses['geom'].item():.4f}{rank_str}"
                    )

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                avg_goal = total_goal / num_batches
                avg_rel = total_rel / num_batches
                avg_geom = total_geom / num_batches
                avg_rank = total_rank / num_batches
                # 计算加权后的 loss 贡献
                weighted_rel = cfg.lambda_rel * avg_rel
                weighted_geom = cfg.lambda_geom * avg_geom
                weighted_rank = getattr(cfg, 'lambda_rank', 0.0) * avg_rank
                rank_str = f" | Rank: {avg_rank:.4f} (λ={getattr(cfg, 'lambda_rank', 0.0)}, weighted={weighted_rank:.4f})" if getattr(cfg, 'lambda_rank', 0.0) > 0 else ""
                print(
                    f"[Train] Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                    f"Goal: {avg_goal:.4f} | Rel: {avg_rel:.4f} (λ={cfg.lambda_rel}, weighted={weighted_rel:.4f}) | "
                    f"Geom: {avg_geom:.4f} (λ={cfg.lambda_geom}, weighted={weighted_geom:.4f}){rank_str}"
                )

            if epoch % cfg.val_interval == 0:
                acc = self.validate(epoch)
                gs3 = getattr(self, 'last_val_gs3', 0.0)  # 从 validate 中获取 GS@3
                
                # 记录每个 epoch 的验证指标到文件
                val_log_path = os.path.join(cfg.save_dir, "val_metrics_per_epoch.txt")
                os.makedirs(os.path.dirname(val_log_path), exist_ok=True)
                # 第一次写入时添加表头
                is_first_write = not os.path.exists(val_log_path)
                with open(val_log_path, 'a', encoding='utf-8') as f:
                    if is_first_write:
                        f.write("Epoch | GS@1 | GS@3\n")
                    f.write(f"{epoch:03d} | {acc * 100:.2f}% | {gs3 * 100:.2f}%\n")
                
                if acc > self.best_val_acc:
                    self.best_val_acc = acc
                    self.best_epoch = epoch
                    # 根据训练列表命名best权重，混合训练集使用 *_mixed 后缀，便于区分
                    # 如果 model_name 包含 "gated"，则使用 gated 后缀
                    is_gated = isinstance(cfg.model_name, str) and "gated" in cfg.model_name.lower()
                    is_mixed = isinstance(cfg.train_list, str) and ("train_mixed_list" in cfg.train_list or "mixed" in cfg.train_list)
                    is_mtl = (getattr(cfg, "lambda_rel", 0.0) > 0.0) or (getattr(cfg, "lambda_geom", 0.0) > 0.0)
                    
                    if is_gated:
                        # Gate 版本的命名
                        if is_mixed and is_mtl:
                            best_name = "best_lc_org_mtl_mixed_gated.pt"
                        elif is_mixed:
                            best_name = "best_lc_org_nomask_mixed_gated.pt"
                        elif is_mtl:
                            best_name = "best_lc_org_mtl_gated.pt"
                        else:
                            best_name = "best_lc_org_nomask_gated.pt"
                    else:
                        # 原版命名
                        if is_mixed and is_mtl:
                            best_name = "best_lc_org_mtl_mixed.pt"
                        elif is_mixed:
                            best_name = "best_lc_org_nomask_mixed.pt"
                        elif is_mtl:
                            best_name = "best_lc_org_mtl.pt"
                        else:
                            best_name = "best_lc_org_nomask.pt"
                    best_path = os.path.join(cfg.save_dir, best_name)
                    torch.save(self.model.state_dict(), best_path)
                    print(f"[Best] Epoch {epoch:03d} | seen_val GS@1: {acc * 100:.2f}% | GS@3: {gs3 * 100:.2f}% | saved: {best_path}")

            # save checkpoint each epoch
            ckpt_path = os.path.join(cfg.save_dir, f"epoch_{epoch:03d}_" + cfg.model_name)
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved to {ckpt_path}")
            import sys
            sys.stdout.flush()  # 确保输出立即刷新
        
        # 训练完成信息
        import sys
        print("\n" + "=" * 80)
        print("训练完成！Training Completed!")
        print("=" * 80)
        print(f"总训练轮数: {cfg.num_epochs}")
        print(f"最佳验证准确率: {self.best_val_acc * 100:.2f}% (Epoch {self.best_epoch:03d})")
        
        # 确定最佳模型路径（使用与保存时相同的逻辑）
        is_gated = isinstance(cfg.model_name, str) and "gated" in cfg.model_name.lower()
        is_mixed = isinstance(cfg.train_list, str) and ("train_mixed_list" in cfg.train_list or "mixed" in cfg.train_list)
        is_mtl = (getattr(cfg, "lambda_rel", 0.0) > 0.0) or (getattr(cfg, "lambda_geom", 0.0) > 0.0)
        
        if is_gated:
            if is_mixed and is_mtl:
                best_name = "best_lc_org_mtl_mixed_gated.pt"
            elif is_mixed:
                best_name = "best_lc_org_nomask_mixed_gated.pt"
            elif is_mtl:
                best_name = "best_lc_org_mtl_gated.pt"
            else:
                best_name = "best_lc_org_nomask_gated.pt"
        else:
            if is_mixed and is_mtl:
                best_name = "best_lc_org_mtl_mixed.pt"
            elif is_mixed:
                best_name = "best_lc_org_nomask_mixed.pt"
            elif is_mtl:
                best_name = "best_lc_org_mtl.pt"
            else:
                best_name = "best_lc_org_nomask.pt"
        
        best_path = os.path.join(cfg.save_dir, best_name)
        if os.path.exists(best_path):
            print(f"最佳模型已保存至: {best_path}")
        print(f"所有检查点保存在: {cfg.save_dir}")
        print("=" * 80 + "\n")
        sys.stdout.flush()

    def validate(self, epoch: int):
        cfg = self.cfg
        self.model.eval()
        correct_top1 = 0
        correct_top3 = 0
        total = 0
        with torch.no_grad():
            for sample in self.val_loader:
                sample = _move_to_device(sample, self.device)
                outputs = self.model(
                    node_feats=sample["node_feats"],
                    edge_index=sample["edge_index"],
                    edge_attr=sample["edge_attr"],
                    lang_feat=sample["lang_feat"],
                    candidate_mask=sample["candidate_mask"],
                )
                goal_logits = outputs["goal_logits"]
                if goal_logits.dim() == 1:
                    goal_logits = goal_logits.unsqueeze(0)

                B = goal_logits.size(0)
                for b in range(B):
                    logits_nodes = goal_logits[b]
                    candidate_mask = sample["candidate_mask"][b] if sample["candidate_mask"].dim() == 2 else sample["candidate_mask"]
                    target_index = sample["target_index"][b].item() if isinstance(sample["target_index"], torch.Tensor) and sample["target_index"].dim() > 0 else sample["target_index"]

                    cand_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
                    if cand_indices.numel() == 0:
                        continue
                    logits_cand = logits_nodes[cand_indices]
                    # 获取 top-3
                    _, sorted_indices = torch.sort(logits_cand, descending=True)
                    top3_local = sorted_indices[:3]
                    top3_global = [int(cand_indices[idx].item()) for idx in top3_local]
                    
                    # GS@1
                    if top3_global and top3_global[0] == int(target_index):
                        correct_top1 += 1
                    # GS@3
                    if int(target_index) in top3_global:
                        correct_top3 += 1
                    total += 1

        acc = correct_top1 / total if total > 0 else 0.0
        gs3 = correct_top3 / total if total > 0 else 0.0
        self.last_val_gs3 = gs3  # 保存 GS@3 供训练循环使用
        print(f"[Val] Epoch {epoch:03d} | GS@1: {acc * 100:.2f}% | GS@3: {gs3 * 100:.2f}% (N={total})")
        return acc
