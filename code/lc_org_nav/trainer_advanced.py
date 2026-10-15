"""
Advanced trainer with state-of-the-art training strategies:

1. Learning rate scheduling (cosine annealing with warmup)
2. Gradient clipping and accumulation
3. Exponential Moving Average (EMA) of model weights
4. Mixed precision training (AMP)
5. Early stopping
6. Comprehensive logging and checkpointing
7. TensorBoard integration
"""

import os
import math
import copy
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from .config import TrainConfig
from .dataset import GoalSelectionDataset
from .model_advanced import LCORGNetAdvanced
from .losses import CombinedLoss
from .metrics import MetricsCalculator, compute_batch_metrics
from .visualization import SceneGraphVisualizer
from .inference import export_to_torchscript


def _move_to_device(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move sample tensors to device."""
    out: Dict[str, Any] = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


class EMA:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine annealing."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, 
                 min_lr_ratio: float = 0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, current_step: int):
        """Update learning rate."""
        if current_step < self.warmup_steps:
            # Linear warmup
            lr_mult = current_step / max(1, self.warmup_steps)
        else:
            # Cosine annealing
            progress = (current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr_mult = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_mult
        
        return lr_mult


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """Check if training should stop.
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


class AdvancedTrainer:
    """Advanced trainer with state-of-the-art training strategies."""
    
    def __init__(self, cfg: TrainConfig, use_advanced_model: bool = True):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        
        # Datasets
        self.train_ds = GoalSelectionDataset(cfg.train_list)
        self.val_ds = GoalSelectionDataset(cfg.val_list)
        
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
        )
        
        # Infer feature dimensions
        sample0 = self.train_ds[0]
        node_feat_dim = sample0["node_feats"].shape[1]
        edge_feat_dim = sample0["edge_attr"].shape[1]
        lang_feat_dim = sample0["lang_feat"].shape[0]
        
        # Model
        if use_advanced_model:
            self.model = LCORGNetAdvanced(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                lang_feat_dim=lang_feat_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                num_rel_types=cfg.num_rel_types,
                num_heads=getattr(cfg, 'num_heads', 4),
                dropout=getattr(cfg, 'dropout', 0.1),
                use_edge_dropout=getattr(cfg, 'use_edge_dropout', True),
                edge_dropout_rate=getattr(cfg, 'edge_dropout_rate', 0.1),
            ).to(self.device)
        else:
            from .model import LCORGNet
            self.model = LCORGNet(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                lang_feat_dim=lang_feat_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                num_rel_types=cfg.num_rel_types,
            ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Loss function
        self.criterion = CombinedLoss(
            lambda_goal=getattr(cfg, 'lambda_goal', 1.0),
            lambda_rel=getattr(cfg, 'lambda_rel', 0.5),
            lambda_geom=getattr(cfg, 'lambda_geom', 0.1),
            lambda_contrastive=getattr(cfg, 'lambda_contrastive', 0.1),
            lambda_hard_neg=getattr(cfg, 'lambda_hard_neg', 0.05),
            use_focal=getattr(cfg, 'use_focal', True),
            use_contrastive=getattr(cfg, 'use_contrastive', True),
            use_uncertainty=getattr(cfg, 'use_uncertainty', True),
            use_hard_neg=getattr(cfg, 'use_hard_neg', True),
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * cfg.num_epochs
        warmup_steps = int(total_steps * getattr(cfg, 'warmup_ratio', 0.1))
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=getattr(cfg, 'min_lr_ratio', 0.01),
        )
        
        # Gradient scaler for mixed precision
        self.use_amp = getattr(cfg, 'use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # EMA
        self.use_ema = getattr(cfg, 'use_ema', True)
        self.ema = EMA(self.model, decay=getattr(cfg, 'ema_decay', 0.999)) if self.use_ema else None
        
        # Gradient clipping
        self.max_grad_norm = getattr(cfg, 'max_grad_norm', 1.0)
        
        # Gradient accumulation
        self.grad_accum_steps = getattr(cfg, 'grad_accum_steps', 1)
        
        # Early stopping
        self.use_early_stopping = getattr(cfg, 'use_early_stopping', True)
        self.early_stopping = EarlyStopping(
            patience=getattr(cfg, 'early_stopping_patience', 10),
            min_delta=getattr(cfg, 'early_stopping_min_delta', 0.001),
            mode='max',
        ) if self.use_early_stopping else None
        
        # Metrics
        self.metrics_calc = MetricsCalculator(
            num_rel_types=cfg.num_rel_types,
            top_k_list=getattr(cfg, 'top_k_list', [1, 3, 5]),
        )
        
        # Logging
        os.makedirs(cfg.save_dir, exist_ok=True)
        base_log_dir = Path(cfg.save_dir) / "logs"
        base_log_dir.mkdir(exist_ok=True)
        # unique run directory with timestamp to avoid overlap across runs
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_log_dir = base_log_dir / f"run-{ts}"
        run_log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(run_log_dir))
        print(f"[Logging] TensorBoard run directory: {run_log_dir}")
        
        # Training state
        self.global_step = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0

        # Reporting / export controls
        self.report_milestones = getattr(cfg, 'report_milestones', [40, 80])
        self.report_num_visualize = getattr(cfg, 'report_num_visualize', 10)
        self.export_on_best = getattr(cfg, 'export_on_best', True)
        self.export_formats = getattr(cfg, 'export_formats', ['torchscript'])
        self.checkpoint_dir = Path(cfg.save_dir)
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        # Aggregate per-epoch training metrics
        est_goal_correct = 0
        est_goal_total = 0
        est_goal_skipped = 0
        loss_components = {
            "goal": 0.0,
            "rel": 0.0,
            "geom": 0.0,
            "contrastive": 0.0,
            "hard_neg": 0.0,
        }
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, sample in enumerate(self.train_loader, start=1):
            sample = _move_to_device(sample, self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                # Get embeddings for contrastive loss if enabled
                return_embeddings = self.criterion.use_contrastive
                
                outputs = self.model(
                    node_feats=sample["node_feats"],
                    edge_index=sample["edge_index"],
                    edge_attr=sample["edge_attr"],
                    lang_feat=sample["lang_feat"],
                    candidate_mask=sample["candidate_mask"],
                    return_embeddings=return_embeddings,
                )
                
                # Compute losses
                losses = self.criterion(outputs, sample, self.device)
                loss = losses["total"] / self.grad_accum_steps
            
            # Estimate GS@1 on-the-fly (only over candidates)
            try:
                goal_logits_b = outputs["goal_logits"]
                if goal_logits_b.dim() == 1:
                    goal_logits_b = goal_logits_b.unsqueeze(0)
                B = goal_logits_b.size(0)
                for bi in range(B):
                    logits_i = goal_logits_b[bi]
                    # candidate mask shape handling
                    cand_mask_i = sample["candidate_mask"][bi] if isinstance(sample["candidate_mask"], torch.Tensor) and sample["candidate_mask"].dim() == 2 else sample["candidate_mask"]
                    cand_indices = torch.nonzero(cand_mask_i, as_tuple=False).squeeze(-1)
                    if cand_indices.numel() == 0:
                        est_goal_skipped += 1
                        continue
                    logits_cand = logits_i[cand_indices]
                    pred_local = int(torch.argmax(logits_cand).item())
                    pred_global = int(cand_indices[pred_local].item())
                    tgt = sample["target_index"][bi].item() if isinstance(sample["target_index"], torch.Tensor) and sample["target_index"].dim() > 0 else int(sample["target_index"])
                    est_goal_correct += int(pred_global == tgt)
                    est_goal_total += 1
            except Exception:
                # Do not break training if estimation fails
                pass

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx % self.grad_accum_steps == 0) or (batch_idx == len(self.train_loader)):
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.use_ema:
                    self.ema.update()
                
                # Update learning rate
                self.scheduler.step(self.global_step)
                self.global_step += 1
            
            # Accumulate losses
            total_loss += losses["total"].item()
            for k in loss_components:
                if k in losses:
                    loss_components[k] += losses[k].item()
            num_batches += 1
            
            # Logging
            if batch_idx % self.cfg.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch:03d} | Batch {batch_idx:04d}/{len(self.train_loader)} | "
                    f"Loss: {losses['total'].item():.4f} | "
                    f"Goal: {losses['goal'].item():.4f} | "
                    f"Rel: {losses['rel'].item():.4f} | "
                    f"Geom: {losses['geom'].item():.4f} | "
                    f"LR: {current_lr:.6f}"
                )
                
                # TensorBoard logging
                self.writer.add_scalar("train/loss", losses["total"].item(), self.global_step)
                self.writer.add_scalar("train/lr", current_lr, self.global_step)
                for k, v in losses.items():
                    if k != "total":
                        self.writer.add_scalar(f"train/loss_{k}", v.item(), self.global_step)
        
        # Average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        for k in loss_components:
            loss_components[k] = loss_components[k] / num_batches if num_batches > 0 else 0.0
        
        # Aggregate estimated GS@1 over epoch
        est_acc = (est_goal_correct / est_goal_total) if est_goal_total > 0 else 0.0
        if est_goal_total == 0 and est_goal_skipped > 0:
            print(f"[Warn] Epoch {epoch:03d}: No valid candidate samples for GS@1 estimation (skipped={est_goal_skipped}).")
        else:
            print(f"[Train][Est] Epoch {epoch:03d} GS@1 over candidates: {est_acc*100:.2f}% ({est_goal_correct}/{est_goal_total}, skipped={est_goal_skipped})")
        # Log epoch-level estimated GS@1
        try:
            self.writer.add_scalar("train/epoch_gs1_est", est_acc, epoch)
        except Exception:
            pass

        return {"total": avg_loss, **loss_components, "gs1_est": est_acc}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        # Apply EMA if enabled
        if self.use_ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        self.metrics_calc.reset()
        
        with torch.no_grad():
            for sample in self.val_loader:
                sample = _move_to_device(sample, self.device)
                
                outputs = self.model(
                    node_feats=sample["node_feats"],
                    edge_index=sample["edge_index"],
                    edge_attr=sample["edge_attr"],
                    lang_feat=sample["lang_feat"],
                    candidate_mask=sample["candidate_mask"],
                    return_embeddings=False,
                )
                
                # Update metrics
                compute_batch_metrics(outputs, sample, self.metrics_calc)
        
        # Compute metrics
        metrics = self.metrics_calc.compute_metrics()
        
        # Print metrics
        self.metrics_calc.print_metrics(prefix=f"[Epoch {epoch:03d}]")
        
        # TensorBoard logging
        for k, v in metrics.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)
        
        # Restore original parameters if using EMA
        if self.use_ema:
            self.ema.restore()
        
        return metrics

    def _export_best_model(self, epoch: int, is_best: bool, checkpoint_path: Path):
        if not (self.export_on_best and is_best):
            return
        export_dir = self.checkpoint_dir / "exported"
        export_dir.mkdir(exist_ok=True, parents=True)
        model_name_stem = checkpoint_path.stem
        # TorchScript export
        if 'torchscript' in self.export_formats:
            ts_path = export_dir / f"{model_name_stem}.ts.pt"
            try:
                self.model.eval()
                export_to_torchscript(self.model, str(ts_path))
                print(f"[Export] TorchScript best model saved to {ts_path}")
            except Exception as e:
                print(f"[Export][WARN] TorchScript export failed: {e}")

    def _generate_visual_report(self, epoch: int):
        """Generate a small visual report at specific epochs."""
        try:
            vis = SceneGraphVisualizer(save_dir=f"outputs/epoch_reports/epoch_{epoch:03d}")
            self.model.eval()
            with torch.no_grad():
                count = 0
                for sample in self.val_loader:
                    if count >= self.report_num_visualize:
                        break
                    sample = _move_to_device(sample, self.device)
                    outputs = self.model(
                        node_feats=sample["node_feats"],
                        edge_index=sample["edge_index"],
                        edge_attr=sample["edge_attr"],
                        lang_feat=sample["lang_feat"],
                        candidate_mask=sample["candidate_mask"],
                        return_embeddings=False,
                    )
                    goal_logits = outputs["goal_logits"]
                    if goal_logits.dim() == 2:
                        goal_logits_b = goal_logits[0]
                    else:
                        goal_logits_b = goal_logits
                    candidate_mask = sample["candidate_mask"][0] if sample["candidate_mask"].dim() == 2 else sample["candidate_mask"]
                    cand_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
                    pred_global = -1
                    if cand_indices.numel() > 0:
                        logits_cand = goal_logits_b[cand_indices]
                        pred_local = int(torch.argmax(logits_cand).item())
                        pred_global = int(cand_indices[pred_local].item())
                    # Prepare numpy arrays, squeeze batch if present
                    node_np = sample["node_feats"].detach().cpu().numpy()
                    edge_np = sample["edge_index"].detach().cpu().numpy()
                    cand_np = sample["candidate_mask"].detach().cpu().numpy()
                    if node_np.ndim == 3:
                        node_np = node_np[0]
                    if edge_np.ndim == 3:
                        edge_np = edge_np[0]
                    if cand_np.ndim == 2:
                        cand_np = cand_np[0]
                    goal_scores_np = goal_logits_b.detach().cpu().numpy()
                    target_index = sample["target_index"][0].item() if isinstance(sample["target_index"], torch.Tensor) and sample["target_index"].dim() > 0 else int(sample["target_index"])
                    vis.visualize_scene_graph(
                        node_feats=node_np,
                        edge_index=edge_np,
                        candidate_mask=cand_np,
                        target_index=target_index,
                        predicted_index=pred_global,
                        goal_scores=goal_scores_np,
                        save_name=f"scene_{count:04d}.png",
                    )
                    count += 1
            print(f"[Report] Visual report generated for epoch {epoch} at outputs/epoch_reports/epoch_{epoch:03d}")
        except Exception as e:
            print(f"[Report][WARN] Failed to generate visual report at epoch {epoch}: {e}")
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.cfg,
        }
        
        if self.use_ema:
            checkpoint["ema_shadow"] = self.ema.shadow
        
        # Save regular checkpoint
        ckpt_path = Path(self.cfg.save_dir) / f"epoch_{epoch:03d}_{self.cfg.model_name}"
        torch.save(checkpoint, ckpt_path)
        print(f"[Checkpoint] Saved to {ckpt_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.cfg.save_dir) / f"best_{self.cfg.model_name}"
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] Best model saved to {best_path}")
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*80}")
        print(f"Training Configuration")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Train samples: {len(self.train_ds)}")
        print(f"Val samples: {len(self.val_ds)}")
        print(f"Epochs: {self.cfg.num_epochs}")
        print(f"Batch size: {self.cfg.batch_size}")
        print(f"Learning rate: {self.cfg.lr}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"EMA: {self.use_ema}")
        print(f"Gradient accumulation steps: {self.grad_accum_steps}")
        print(f"{'='*80}\n")
        
        for epoch in range(1, self.cfg.num_epochs + 1):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{self.cfg.num_epochs}")
            print(f"{'='*80}")
            
            # Train
            epoch_start_time = time.perf_counter()
            train_losses = self.train_epoch(epoch)
            epoch_time_sec = time.perf_counter() - epoch_start_time
            print(f"\n[Train] Epoch {epoch:03d} | Loss: {train_losses['total']:.4f}")
            print(f"[Timing] Epoch {epoch:03d} time: {epoch_time_sec:.2f} sec")
            # Log per-epoch aggregated losses with epoch as x-axis
            try:
                self.writer.add_scalar("train/epoch_total", train_losses.get("total", 0.0), epoch)
                self.writer.add_scalar("train/epoch_goal", train_losses.get("goal", 0.0), epoch)
                self.writer.add_scalar("train/epoch_rel", train_losses.get("rel", 0.0), epoch)
                self.writer.add_scalar("train/epoch_geom", train_losses.get("geom", 0.0), epoch)
                self.writer.add_scalar("train/epoch_contrastive", train_losses.get("contrastive", 0.0), epoch)
                self.writer.add_scalar("train/epoch_hard_neg", train_losses.get("hard_neg", 0.0), epoch)
                self.writer.add_scalar("train/epoch_time_sec", epoch_time_sec, epoch)
            except Exception:
                pass
            
            # Validate
            if epoch % self.cfg.val_interval == 0:
                val_start = time.perf_counter()
                val_metrics = self.validate(epoch)
                val_time = time.perf_counter() - val_start
                print(f"[Timing] Epoch {epoch:03d} validation time: {val_time:.2f} sec")
                
                # Check if best model
                val_acc = val_metrics.get("goal_top1_acc", 0.0)
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    print(f"\n[Best] New best validation accuracy: {val_acc:.2f}% at epoch {epoch}")
                
                # Save checkpoint (only on best or at fixed intervals)
                should_save_regular = (epoch % getattr(self.cfg, 'save_interval', 5) == 0)
                if is_best or should_save_regular:
                    save_start = time.perf_counter()
                    self.save_checkpoint(epoch, val_metrics, is_best)
                    save_time = time.perf_counter() - save_start
                    print(f"[Timing] Epoch {epoch:03d} checkpoint save time: {save_time:.2f} sec")

                # Export best model (TorchScript) if configured
                ckpt_path = Path(self.cfg.save_dir) / f"epoch_{epoch:03d}_{self.cfg.model_name}"
                self._export_best_model(epoch, is_best, ckpt_path)

                # Generate visual report at milestones
                if epoch in self.report_milestones:
                    self._generate_visual_report(epoch)
                
                # Early stopping
                if self.use_early_stopping:
                    if self.early_stopping(val_acc):
                        print(f"\n[Early Stopping] No improvement for {self.early_stopping.patience} epochs.")
                        print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
                        break
        
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        print(f"{'='*80}\n")
        
        self.writer.close()

