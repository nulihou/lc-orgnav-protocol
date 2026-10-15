"""
Comprehensive evaluation metrics for LC-ORGNav.

Includes:
1. Top-K accuracy for goal selection
2. Mean Average Precision (mAP)
3. Geometric error metrics (distance, angle)
4. Relation classification metrics (precision, recall, F1)
5. Calibration metrics for uncertainty
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


class MetricsCalculator:
    """Comprehensive metrics calculator for goal selection task."""
    
    def __init__(self, num_rel_types: int = 5, top_k_list: List[int] = [1, 3, 5]):
        self.num_rel_types = num_rel_types
        self.top_k_list = top_k_list
        self.reset()
        
    def reset(self):
        """Reset all accumulated metrics."""
        # Goal selection metrics
        self.goal_correct = {k: 0 for k in self.top_k_list}
        self.goal_total = 0
        
        # Relation classification metrics
        self.rel_correct = 0
        self.rel_total = 0
        self.rel_confusion = np.zeros((self.num_rel_types, self.num_rel_types), dtype=np.int64)
        
        # Geometric metrics
        self.dist_errors = []
        self.angle_errors = []
        self.dist_abs_errors = []
        self.angle_abs_errors = []
        
        # Ranking metrics
        self.average_precisions = []
        self.reciprocal_ranks = []
        
        # Uncertainty metrics (if available)
        self.uncertainties = []
        self.errors_with_uncertainty = []
        
    def update_goal_metrics(self, logits: torch.Tensor, target_index: int,
                           candidate_mask: torch.Tensor):
        """Update goal selection metrics.
        
        Args:
            logits: (N,) per-node logits
            target_index: ground truth target index
            candidate_mask: (N,) boolean mask for candidates
        """
        # Handle batch dimension
        if logits.dim() > 1:
            logits = logits.squeeze(0)
        if candidate_mask.dim() > 1:
            candidate_mask = candidate_mask.squeeze(0)
        
        # Filter to candidates
        cand_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        if cand_indices.dim() == 0:
            cand_indices = cand_indices.unsqueeze(0)
        if cand_indices.numel() == 0:
            return
        
        logits_cand = logits[cand_indices]
        
        # Check if target is in candidates
        matches = torch.nonzero(cand_indices == target_index, as_tuple=False).squeeze(-1)
        if matches.numel() == 0:
            return
        
        target_local = matches[0].item()
        
        # Top-K accuracy
        num_cands = logits_cand.size(0)
        for k in self.top_k_list:
            if k > num_cands:
                continue
            _, top_k_indices = torch.topk(logits_cand, min(k, num_cands))
            if target_local in top_k_indices.cpu().numpy():
                self.goal_correct[k] += 1
        
        self.goal_total += 1
        
        # Average Precision (AP)
        sorted_indices = torch.argsort(logits_cand, descending=True)
        rank = (sorted_indices == target_local).nonzero(as_tuple=False).item() + 1
        ap = 1.0 / rank
        self.average_precisions.append(ap)
        
        # Mean Reciprocal Rank (MRR)
        self.reciprocal_ranks.append(1.0 / rank)
        
    def update_relation_metrics(self, logits: torch.Tensor, target: int):
        """Update relation classification metrics.
        
        Args:
            logits: (num_rel_types,) relation logits
            target: ground truth relation type
        """
        pred = torch.argmax(logits).item()
        target = int(target)
        
        if pred == target:
            self.rel_correct += 1
        self.rel_total += 1
        
        # Update confusion matrix
        self.rel_confusion[target, pred] += 1
        
    def update_geometric_metrics(self, pred: torch.Tensor, target: torch.Tensor,
                                 logvar: torch.Tensor = None):
        """Update geometric prediction metrics.
        
        Args:
            pred: (2,) predicted [distance, angle]
            target: (2,) ground truth [distance, angle]
            logvar: (2,) optional log variance for uncertainty
        """
        pred = pred.detach().cpu()
        target = target.detach().cpu()
        
        dist_pred, angle_pred = pred[0].item(), pred[1].item()
        dist_target, angle_target = target[0].item(), target[1].item()
        
        # Distance errors
        dist_error = dist_pred - dist_target
        dist_abs_error = abs(dist_error)
        self.dist_errors.append(dist_error)
        self.dist_abs_errors.append(dist_abs_error)
        
        # Angle errors (handle periodicity)
        angle_diff = angle_pred - angle_target
        # Wrap to [-pi, pi]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        angle_abs_error = abs(angle_diff)
        self.angle_errors.append(angle_diff)
        self.angle_abs_errors.append(angle_abs_error)
        
        # Uncertainty metrics
        if logvar is not None:
            logvar = logvar.detach().cpu()
            uncertainty = torch.exp(logvar).numpy()
            self.uncertainties.append(uncertainty)
            self.errors_with_uncertainty.append([dist_abs_error, angle_abs_error])
            
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all accumulated metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Goal selection metrics
        if self.goal_total > 0:
            for k in self.top_k_list:
                acc = self.goal_correct[k] / self.goal_total * 100
                metrics[f"goal_top{k}_acc"] = acc
            
            # mAP and MRR
            if self.average_precisions:
                metrics["goal_mAP"] = np.mean(self.average_precisions) * 100
            if self.reciprocal_ranks:
                metrics["goal_MRR"] = np.mean(self.reciprocal_ranks)
        
        # Relation classification metrics
        if self.rel_total > 0:
            metrics["rel_acc"] = self.rel_correct / self.rel_total * 100
            
            # Per-class precision, recall, F1
            precisions = []
            recalls = []
            f1_scores = []
            
            for i in range(self.num_rel_types):
                tp = self.rel_confusion[i, i]
                fp = self.rel_confusion[:, i].sum() - tp
                fn = self.rel_confusion[i, :].sum() - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
            
            metrics["rel_precision_macro"] = np.mean(precisions) * 100
            metrics["rel_recall_macro"] = np.mean(recalls) * 100
            metrics["rel_f1_macro"] = np.mean(f1_scores) * 100
        
        # Geometric metrics
        if self.dist_errors:
            metrics["dist_mae"] = np.mean(self.dist_abs_errors)
            metrics["dist_rmse"] = np.sqrt(np.mean(np.square(self.dist_errors)))
            metrics["dist_median_ae"] = np.median(self.dist_abs_errors)
            
        if self.angle_errors:
            metrics["angle_mae"] = np.mean(self.angle_abs_errors) * 180 / np.pi  # Convert to degrees
            metrics["angle_rmse"] = np.sqrt(np.mean(np.square(self.angle_errors))) * 180 / np.pi
            metrics["angle_median_ae"] = np.median(self.angle_abs_errors) * 180 / np.pi
        
        # Uncertainty calibration (if available)
        if self.uncertainties and self.errors_with_uncertainty:
            uncertainties = np.array(self.uncertainties)  # (N, 2)
            errors = np.array(self.errors_with_uncertainty)  # (N, 2)
            
            # Compute correlation between uncertainty and error
            dist_corr = np.corrcoef(uncertainties[:, 0], errors[:, 0])[0, 1]
            angle_corr = np.corrcoef(uncertainties[:, 1], errors[:, 1])[0, 1]
            
            metrics["dist_uncertainty_corr"] = dist_corr
            metrics["angle_uncertainty_corr"] = angle_corr
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get relation classification confusion matrix."""
        return self.rel_confusion.copy()
    
    def print_metrics(self, prefix: str = ""):
        """Print formatted metrics."""
        metrics = self.compute_metrics()
        
        print(f"\n{'='*60}")
        print(f"{prefix} Evaluation Metrics")
        print(f"{'='*60}")
        
        # Goal selection
        print("\n[Goal Selection]")
        for k in self.top_k_list:
            key = f"goal_top{k}_acc"
            if key in metrics:
                print(f"  Top-{k} Accuracy: {metrics[key]:.2f}%")
        if "goal_mAP" in metrics:
            print(f"  mAP: {metrics['goal_mAP']:.2f}%")
        if "goal_MRR" in metrics:
            print(f"  MRR: {metrics['goal_MRR']:.4f}")
        
        # Relation classification
        print("\n[Relation Classification]")
        if "rel_acc" in metrics:
            print(f"  Accuracy: {metrics['rel_acc']:.2f}%")
            print(f"  Precision (macro): {metrics['rel_precision_macro']:.2f}%")
            print(f"  Recall (macro): {metrics['rel_recall_macro']:.2f}%")
            print(f"  F1 (macro): {metrics['rel_f1_macro']:.2f}%")
        
        # Geometric prediction
        print("\n[Geometric Prediction]")
        if "dist_mae" in metrics:
            print(f"  Distance MAE: {metrics['dist_mae']:.2f}m")
            print(f"  Distance RMSE: {metrics['dist_rmse']:.2f}m")
            print(f"  Distance Median AE: {metrics['dist_median_ae']:.2f}m")
        if "angle_mae" in metrics:
            print(f"  Angle MAE: {metrics['angle_mae']:.2f}°")
            print(f"  Angle RMSE: {metrics['angle_rmse']:.2f}°")
            print(f"  Angle Median AE: {metrics['angle_median_ae']:.2f}°")
        
        # Uncertainty calibration
        if "dist_uncertainty_corr" in metrics:
            print("\n[Uncertainty Calibration]")
            print(f"  Distance Uncertainty-Error Correlation: {metrics['dist_uncertainty_corr']:.3f}")
            print(f"  Angle Uncertainty-Error Correlation: {metrics['angle_uncertainty_corr']:.3f}")
        
        print(f"{'='*60}\n")


def compute_batch_metrics(outputs: Dict[str, torch.Tensor], 
                         samples: Dict[str, torch.Tensor],
                         metrics_calc: MetricsCalculator):
    """Compute metrics for a batch of samples.
    
    Args:
        outputs: model outputs
        samples: ground truth samples
        metrics_calc: MetricsCalculator instance to update
    """
    goal_logits = outputs["goal_logits"]
    rel_type_logits = outputs["rel_type_logits"]
    geom_pred = outputs["geom_pred"]
    geom_logvar = outputs.get("geom_logvar", None)
    
    # Handle single sample - don't add batch dim, process as single
    if goal_logits.dim() == 1:
        # Single sample, process directly
        candidate_mask = samples["candidate_mask"]
        if candidate_mask.dim() > 1:
            candidate_mask = candidate_mask.squeeze(0)
        
        target_index = samples["target_index"]
        if isinstance(target_index, torch.Tensor):
            if target_index.dim() > 0:
                target_index = target_index.item()
            else:
                target_index = target_index.item()
        
        rel_type = samples["rel_type"]
        if isinstance(rel_type, torch.Tensor):
            if rel_type.dim() > 0:
                rel_type = rel_type.item()
            else:
                rel_type = rel_type.item()
        
        geom_gt = samples["geom"]
        if geom_gt.dim() > 1:
            geom_gt = geom_gt.squeeze(0)
        
        geom_logvar_single = geom_logvar if geom_logvar is None else geom_logvar
        
        # Update metrics
        metrics_calc.update_goal_metrics(goal_logits, int(target_index), candidate_mask)
        metrics_calc.update_relation_metrics(rel_type_logits, int(rel_type))
        metrics_calc.update_geometric_metrics(geom_pred, geom_gt, geom_logvar_single)
        return
    
    # Batch processing
    if geom_logvar is not None and geom_logvar.dim() == 1:
        geom_logvar = geom_logvar.unsqueeze(0)
    
    B = goal_logits.size(0)
    
    for b in range(B):
        logits_b = goal_logits[b]
        rel_logits_b = rel_type_logits[b]
        geom_pred_b = geom_pred[b]
        geom_logvar_b = geom_logvar[b] if geom_logvar is not None else None
        
        # Extract sample data
        candidate_mask = samples["candidate_mask"][b] if isinstance(samples["candidate_mask"], torch.Tensor) and samples["candidate_mask"].dim() == 2 else samples["candidate_mask"]
        target_index = samples["target_index"][b].item() if isinstance(samples["target_index"], torch.Tensor) and samples["target_index"].dim() > 0 else samples["target_index"]
        rel_type = samples["rel_type"][b].item() if isinstance(samples["rel_type"], torch.Tensor) and samples["rel_type"].dim() > 0 else samples["rel_type"]
        geom_gt = samples["geom"][b] if isinstance(samples["geom"], torch.Tensor) and samples["geom"].dim() == 2 else samples["geom"]
        
        # Update metrics
        metrics_calc.update_goal_metrics(logits_b, int(target_index), candidate_mask)
        metrics_calc.update_relation_metrics(rel_logits_b, int(rel_type))
        metrics_calc.update_geometric_metrics(geom_pred_b, geom_gt, geom_logvar_b)

