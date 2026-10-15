
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Sampler
from typing import List, Dict, Any, Iterator
import math

from .trainer import GoalSelectorTrainer, _move_to_device
from .config import TrainConfig
from .model import LCORGNet

class MixedBatchSampler(Sampler[List[int]]):
    """
    Custom BatchSampler that mixes Easy (K=2) and Hard (K>=4) samples 
    with a fixed ratio in every batch.
    """
    def __init__(self, easy_indices: List[int], hard_indices: List[int], batch_size: int, hard_ratio: float = 0.3):
        self.easy_indices = easy_indices
        self.hard_indices = hard_indices
        self.batch_size = batch_size
        self.hard_count = max(1, int(batch_size * hard_ratio))
        self.easy_count = max(1, batch_size - self.hard_count)
        
        # Calculate number of batches based on easy samples (since they are majority)
        # We want to iterate through all easy samples approximately once per epoch
        self.num_batches = len(easy_indices) // self.easy_count
        
        # If hard samples are too few, they will be repeated (oversampling)
        
    def __len__(self):
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle easy indices for this epoch
        easy_perm = torch.randperm(len(self.easy_indices)).tolist()
        
        # For hard indices, we might need to cycle through them if they are fewer than needed
        total_hard_needed = self.num_batches * self.hard_count
        hard_samples = []
        while len(hard_samples) < total_hard_needed:
            # Shuffle hard indices each time we reuse them to avoid same pattern
            hard_samples.extend(torch.randperm(len(self.hard_indices)).tolist())
        
        hard_samples = hard_samples[:total_hard_needed]
        
        for i in range(self.num_batches):
            batch_easy = [self.easy_indices[idx] for idx in easy_perm[i * self.easy_count : (i + 1) * self.easy_count]]
            batch_hard = hard_samples[i * self.hard_count : (i + 1) * self.hard_count]
            yield batch_easy + batch_hard

def mixed_collate_fn(batch):
    """
    Custom collate function for list of variable-sized graph samples.
    """
    elem = batch[0]
    collated = {}
    
    # Fields that are variable sized and should remain as lists
    variable_fields = {'node_feats', 'edge_index', 'edge_attr', 'candidate_mask', 'geom'}
    
    for key in elem:
        if key in variable_fields:
            collated[key] = [d[key] for d in batch]
        else:
            # Use default stacking for fixed size fields (like lang_feat, target_index)
            try:
                collated[key] = torch.stack([d[key] for d in batch], dim=0)
            except Exception:
                    # Fallback for non-stackable types
                collated[key] = [d[key] for d in batch]
                
    return collated

class MixedBatchTrainer(GoalSelectorTrainer):
    def __init__(self, cfg: TrainConfig):
        # Call super init to setup datasets and model
        super().__init__(cfg)
        
        # Re-initialize val_loader with custom collate_fn to handle variable sized graphs
        self.val_loader = DataLoader(
            self.val_ds, 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=0,
            collate_fn=mixed_collate_fn
        )
        
        # Setup Distillation Teacher if configured
        self.teacher_model = None
        self.distill_lambda = getattr(cfg, 'distill_lambda', 0.0)
        
        if self.distill_lambda > 0.0:
            print(f"[Trainer] Distillation enabled (lambda={self.distill_lambda}). Loading teacher...")
            # Assuming finetune_from points to the teacher model or we have a separate config
            # Use 'finetune_from' as teacher path if not explicitly provided as 'teacher_path'
            teacher_path = getattr(cfg, 'teacher_path', getattr(cfg, 'finetune_from', ''))
            
            if teacher_path and os.path.exists(teacher_path):
                self.teacher_model = LCORGNet(
                    node_feat_dim=self.model.node_encoder[0].in_features if hasattr(self.model, 'node_encoder') else 32,
                    edge_feat_dim=self.model.edge_encoder[0].in_features if hasattr(self.model, 'edge_encoder') else 8,
                    lang_feat_dim=self.model.lang_proj[0].in_features if hasattr(self.model, 'lang_proj') else 32,
                    hidden_dim=cfg.hidden_dim,
                    num_layers=cfg.num_layers,
                    num_rel_types=cfg.num_rel_types,
                    use_edge_context_head=getattr(cfg, 'use_edge_context_head', False),
                    use_rel_gate=getattr(cfg, 'use_rel_gate', False),
                ).to(self.device)
                
                # Load weights
                state_dict = torch.load(teacher_path, map_location=self.device)
                self.teacher_model.load_state_dict(state_dict, strict=False)
                self.teacher_model.eval()
                # Freeze teacher
                for param in self.teacher_model.parameters():
                    param.requires_grad = False
                print(f"[Trainer] Teacher model loaded from {teacher_path}")
            else:
                print(f"[Trainer] Warning: Distillation enabled but teacher path invalid: {teacher_path}")
                self.distill_lambda = 0.0

        # Setup Mixed Batch Sampling
        # Check if we should use MixedBatchSampler (e.g. if hard_ratio is set)
        self.hard_ratio = getattr(cfg, 'hard_ratio', 0.0)
        if self.hard_ratio > 0.0:
            print(f"[Trainer] Initializing MixedBatchSampler (Hard Ratio={self.hard_ratio})...")
            high_k_thresh = getattr(cfg, 'high_k_threshold', 4)
            
            easy_indices = []
            hard_indices = []
            
            print(f"[Trainer] Scanning {len(self.train_ds.paths)} files for K>={high_k_thresh}...")
            for i, p in enumerate(self.train_ds.paths):
                if i % 1000 == 0:
                    print(f"  scanned {i} files...")
                try:
                    data = np.load(p, allow_pickle=True)
                    k = 0
                    if 'candidate_mask' in data:
                        cm = data['candidate_mask']
                        if cm.ndim == 2 and cm.shape[0] == 1:
                            cm = cm[0]
                        k = int(np.sum(cm))
                    elif 'num_candidates' in data:
                        k = int(data['num_candidates'])
                    
                    if k >= high_k_thresh:
                        hard_indices.append(i)
                    else:
                        easy_indices.append(i)
                except Exception:
                    easy_indices.append(i) # Default to easy if error
            
            print(f"[Trainer] Split complete. Easy (K<{high_k_thresh}): {len(easy_indices)}, Hard (K>={high_k_thresh}): {len(hard_indices)}")
            
            # Subsample easy_indices for faster sweep
            MAX_EASY = 1000
            if len(easy_indices) > MAX_EASY:
                import random
                random.shuffle(easy_indices)
                easy_indices = easy_indices[:MAX_EASY]
                print(f"[Trainer] Subsampled Easy samples to {MAX_EASY} for efficiency.")

            if len(hard_indices) > 0 and len(easy_indices) > 0:
                batch_sampler = MixedBatchSampler(
                    easy_indices, hard_indices, 
                    batch_size=cfg.batch_size, 
                    hard_ratio=self.hard_ratio
                )
                
                # Re-create train_loader with BatchSampler
                # Note: shuffle must be False when batch_sampler is provided
                self.train_loader = DataLoader(
                    self.train_ds,
                    batch_sampler=batch_sampler,
                    num_workers=0,
                    collate_fn=mixed_collate_fn
                )
                print(f"[Trainer] MixedBatchSampler hooked into DataLoader.")
            else:
                print("[Trainer] Warning: Not enough samples to split. Falling back to default loader.")
                self.train_loader = DataLoader(
                    self.train_ds,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    num_workers=0,
                    collate_fn=mixed_collate_fn
                )
        else:
             # Even without mixed sampler, we need custom collate for batching
             self.train_loader = DataLoader(
                self.train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=mixed_collate_fn
            )

    def compute_losses(self, outputs: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Get base losses
        losses = super().compute_losses(outputs, sample)
        
        # Add Distillation Loss if enabled
        if self.teacher_model is not None and self.distill_lambda > 0.0:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    node_feats=sample["node_feats"],
                    edge_index=sample["edge_index"],
                    edge_attr=sample["edge_attr"],
                    lang_feat=sample["lang_feat"],
                    candidate_mask=sample["candidate_mask"],
                )
            
            student_logits = outputs["goal_logits"]
            teacher_logits = teacher_outputs["goal_logits"]
            
            # Identify K=2 samples (Easy) to apply distillation
            # The user said: "K>=4: normal CE; K=2: CE + lambda * KL"
            # We can check K for each sample in batch
            
            distill_loss = torch.tensor(0.0, device=self.device)
            count = 0
            
            if student_logits.dim() == 1:
                student_logits = student_logits.unsqueeze(0)
                teacher_logits = teacher_logits.unsqueeze(0)
            
            B = student_logits.size(0)
            for b in range(B):
                candidate_mask = sample["candidate_mask"][b]
                k = int(candidate_mask.sum().item())
                high_k_thresh = getattr(self.cfg, 'high_k_threshold', 4)
                
                # Apply distill only if K < high_k_thresh (Easy samples)
                if k < high_k_thresh:
                    cand_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
                    if cand_indices.numel() > 0:
                        s_logits = student_logits[b][cand_indices]
                        t_logits = teacher_logits[b][cand_indices]
                        
                        # KL Divergence
                        # F.kl_div expects input as log-probs, target as probs (or log-probs depending on reduction)
                        # default: input log_softmax, target softmax (if log_target=False)
                        
                        loss_kl = F.kl_div(
                            F.log_softmax(s_logits, dim=0),
                            F.softmax(t_logits, dim=0),
                            reduction='sum' # Sum over classes, mean over batch later
                        )
                        distill_loss += loss_kl
                        count += 1
            
            if count > 0:
                distill_loss = distill_loss / count
                losses["distill"] = distill_loss
                losses["total"] = losses.get("total", 0.0) + self.distill_lambda * distill_loss
            else:
                 losses["distill"] = torch.tensor(0.0, device=self.device)

        return losses

    def train(self):
        # Override train to include distill loss in total loss logging and backward
        # But GoalSelectorTrainer.train() calculates total loss inside itself:
        # loss = losses["goal"] + ...
        # So I need to copy the train loop or inject the loss?
        # GoalSelectorTrainer.train() computes loss explicitly:
        # loss = losses["goal"] + cfg.lambda_rel * losses["rel"] ...
        # It doesn't use a "total" key from compute_losses.
        
        # So I MUST override train() completely or make GoalSelectorTrainer more flexible.
        # Since I cannot modify GoalSelectorTrainer easily without editing the file, 
        # I will copy the train method here and modify it.
        
        cfg = self.cfg
        for epoch in range(1, cfg.num_epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_goal = 0.0
            total_rel = 0.0
            total_geom = 0.0
            total_rank = 0.0
            total_distill = 0.0
            num_batches = 0

            for batch_idx, sample in enumerate(self.train_loader, start=1):
                sample = _move_to_device(sample, self.device)
                sample = self._maybe_override_candidate_mask(sample)

                outputs = self.model(
                    node_feats=sample["node_feats"],
                    edge_index=sample["edge_index"],
                    edge_attr=sample["edge_attr"],
                    lang_feat=sample["lang_feat"],
                    candidate_mask=sample["candidate_mask"],
                )

                losses = self.compute_losses(outputs, sample)
                
                # Calculate total loss
                loss = (
                    losses["goal"]
                    + cfg.lambda_rel * losses["rel"]
                    + cfg.lambda_geom * losses["geom"]
                    + getattr(cfg, 'lambda_rank', 0.0) * losses["rank"]
                )
                
                # Add distill loss if present
                if "distill" in losses:
                    loss += self.distill_lambda * losses["distill"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_goal += losses["goal"].item()
                total_rel += losses["rel"].item()
                total_geom += losses["geom"].item()
                total_rank += losses["rank"].item()
                if "distill" in losses:
                    total_distill += losses["distill"].item()
                    
                num_batches += 1

                if batch_idx % cfg.log_interval == 0:
                    rank_str = f" | Rank: {losses['rank'].item():.4f}" if getattr(cfg, 'lambda_rank', 0.0) > 0 else ""
                    distill_str = f" | Distill: {losses['distill'].item():.4f}" if "distill" in losses else ""
                    print(
                        f"Epoch {epoch:03d} | Batch {batch_idx:04d} | "
                        f"Loss: {loss.item():.4f} | Goal: {losses['goal'].item():.4f} | "
                        f"Rel: {losses['rel'].item():.4f} | Geom: {losses['geom'].item():.4f}"
                        f"{rank_str}{distill_str}"
                    )

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                avg_goal = total_goal / num_batches
                avg_rel = total_rel / num_batches
                avg_geom = total_geom / num_batches
                avg_rank = total_rank / num_batches
                avg_distill = total_distill / num_batches
                
                weighted_rel = cfg.lambda_rel * avg_rel
                weighted_geom = cfg.lambda_geom * avg_geom
                weighted_rank = getattr(cfg, 'lambda_rank', 0.0) * avg_rank
                weighted_distill = self.distill_lambda * avg_distill
                
                rank_str = f" | Rank: {avg_rank:.4f}" if getattr(cfg, 'lambda_rank', 0.0) > 0 else ""
                distill_str = f" | Distill: {avg_distill:.4f} (λ={self.distill_lambda}, w={weighted_distill:.4f})" if self.distill_lambda > 0 else ""
                
                print(
                    f"[Train] Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                    f"Goal: {avg_goal:.4f} | Rel: {avg_rel:.4f} (λ={cfg.lambda_rel}) | "
                    f"Geom: {avg_geom:.4f} (λ={cfg.lambda_geom}){rank_str}{distill_str}"
                )

            if epoch % cfg.val_interval == 0:
                # Reuse validation logic from parent
                acc = self.validate(epoch)
                gs3 = getattr(self, 'last_val_gs3', 0.0)
                
                val_log_path = os.path.join(cfg.save_dir, "val_metrics_per_epoch.txt")
                os.makedirs(os.path.dirname(val_log_path), exist_ok=True)
                is_first_write = not os.path.exists(val_log_path)
                with open(val_log_path, 'a', encoding='utf-8') as f:
                    if is_first_write:
                        f.write("Epoch | GS@1 | GS@3\n")
                    f.write(f"{epoch:03d} | {acc * 100:.2f}% | {gs3 * 100:.2f}%\n")
                
                if acc > self.best_val_acc:
                    self.best_val_acc = acc
                    self.best_epoch = epoch
                    best_name = "best_lc_org_mixed_batch.pt"
                    best_path = os.path.join(cfg.save_dir, best_name)
                    torch.save(self.model.state_dict(), best_path)
                    print(f"[Best] Epoch {epoch:03d} | seen_val GS@1: {acc * 100:.2f}% | GS@3: {gs3 * 100:.2f}% | saved: {best_path}")

            ckpt_path = os.path.join(cfg.save_dir, f"epoch_{epoch:03d}_model.pt")
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved to {ckpt_path}")
            import sys
            sys.stdout.flush()

        print("\n" + "=" * 80)
        print("Training Completed!")
        print("=" * 80)
        print(f"Total Epochs: {cfg.num_epochs}")
        print(f"Best Val Acc: {self.best_val_acc * 100:.2f}% (Epoch {self.best_epoch:03d})")

    def validate(self, epoch: int) -> float:
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
                    
                    # Handle list vs tensor for candidate_mask
                    cm_batch = sample["candidate_mask"]
                    if isinstance(cm_batch, list):
                        candidate_mask = cm_batch[b]
                    elif isinstance(cm_batch, torch.Tensor):
                        candidate_mask = cm_batch[b] if cm_batch.dim() == 2 else cm_batch
                    else:
                        candidate_mask = cm_batch

                    # Handle list vs tensor for target_index
                    ti_batch = sample["target_index"]
                    if isinstance(ti_batch, torch.Tensor):
                         target_index = ti_batch[b].item()
                    elif isinstance(ti_batch, list):
                        target_index = ti_batch[b]
                        if isinstance(target_index, torch.Tensor):
                            target_index = target_index.item()
                    else:
                        target_index = ti_batch

                    cand_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
                    if cand_indices.numel() == 0:
                        continue
                    logits_cand = logits_nodes[cand_indices]
                    # Get top-3
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
        self.last_val_gs3 = gs3
        print(f"[Val] Epoch {epoch:03d} | GS@1: {acc * 100:.2f}% | GS@3: {gs3 * 100:.2f}% (N={total})")
        return acc
