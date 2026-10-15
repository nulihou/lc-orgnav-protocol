"""
Advanced loss functions for LC-ORGNav training.

Includes:
1. Focal loss for handling class imbalance
2. Contrastive loss for representation learning
3. Periodic angle loss for handling angular predictions
4. Uncertainty-aware geometric loss
5. Hard negative mining
"""

import math
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for language-node alignment."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, node_embeddings: torch.Tensor, lang_embedding: torch.Tensor,
                target_index: int, candidate_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings: (N, D) node embeddings
            lang_embedding: (D,) language embedding
            target_index: index of positive node
            candidate_mask: (N,) mask for candidate nodes
        Returns:
            contrastive loss
        """
        # Normalize embeddings
        node_embeddings = F.normalize(node_embeddings, dim=-1)
        lang_embedding = F.normalize(lang_embedding, dim=-1)
        
        # Compute similarities
        similarities = torch.matmul(node_embeddings, lang_embedding) / self.temperature  # (N,)
        
        # Mask non-candidates
        similarities = similarities * candidate_mask.float() + (1 - candidate_mask.float()) * (-1e9)
        
        # Target is the positive sample
        target_tensor = torch.tensor([target_index], device=similarities.device, dtype=torch.long)
        loss = F.cross_entropy(similarities.unsqueeze(0), target_tensor)
        
        return loss


class PeriodicAngleLoss(nn.Module):
    """Loss function that handles periodic nature of angles."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred_angle: torch.Tensor, target_angle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_angle: predicted angle in radians
            target_angle: target angle in radians
        Returns:
            periodic angle loss
        """
        # Compute angular difference considering periodicity
        diff = pred_angle - target_angle
        # Wrap to [-pi, pi]
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        
        # Smooth L1 loss on wrapped difference
        loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction=self.reduction)
        
        return loss


class UncertaintyWeightedLoss(nn.Module):
    """Uncertainty-weighted loss for geometric predictions.
    
    Based on "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_mean: torch.Tensor, pred_logvar: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_mean: (2,) predicted [distance, angle]
            pred_logvar: (2,) predicted log variance
            target: (2,) target [distance, angle]
        Returns:
            uncertainty-weighted loss
        """
        # Separate distance and angle
        dist_mean, angle_mean = pred_mean[0], pred_mean[1]
        dist_logvar, angle_logvar = pred_logvar[0], pred_logvar[1]
        dist_target, angle_target = target[0], target[1]
        
        # Distance loss (Gaussian)
        dist_var = torch.exp(dist_logvar)
        dist_loss = 0.5 * (torch.square(dist_mean - dist_target) / dist_var + dist_logvar)
        
        # Angle loss (von Mises-like with periodic handling)
        angle_diff = angle_mean - angle_target
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        angle_var = torch.exp(angle_logvar)
        angle_loss = 0.5 * (torch.square(angle_diff) / angle_var + angle_logvar)
        
        return dist_loss + angle_loss


class HardNegativeMiningLoss(nn.Module):
    """Hard negative mining for goal selection."""
    
    def __init__(self, neg_ratio: float = 3.0, margin: float = 0.5):
        super().__init__()
        self.neg_ratio = neg_ratio
        self.margin = margin
        
    def forward(self, logits: torch.Tensor, target_index: int, 
                candidate_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N,) per-node logits
            target_index: index of positive node
            candidate_mask: (N,) mask for candidate nodes
        Returns:
            hard negative mining loss
        """
        N = logits.size(0)
        device = logits.device
        
        # Get candidate indices
        cand_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        if cand_indices.numel() == 0:
            return torch.tensor(0.0, device=device)
        
        # Positive sample
        pos_logit = logits[target_index]
        
        # Negative samples (all candidates except target)
        neg_mask = candidate_mask.clone()
        neg_mask[target_index] = False
        neg_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze(-1)
        
        if neg_indices.numel() == 0:
            return torch.tensor(0.0, device=device)
        
        neg_logits = logits[neg_indices]
        
        # Select hard negatives (highest scoring negatives)
        num_hard = min(int(self.neg_ratio), neg_logits.size(0))
        hard_neg_logits, _ = torch.topk(neg_logits, num_hard)
        
        # Hinge loss: max(0, margin + neg_logit - pos_logit)
        losses = F.relu(self.margin + hard_neg_logits - pos_logit)
        
        return losses.mean()


class CombinedLoss(nn.Module):
    """Combined loss function with multiple objectives."""
    
    def __init__(
        self,
        lambda_goal: float = 1.0,
        lambda_rel: float = 0.5,
        lambda_geom: float = 0.1,
        lambda_contrastive: float = 0.1,
        lambda_hard_neg: float = 0.05,
        use_focal: bool = True,
        use_contrastive: bool = True,
        use_uncertainty: bool = True,
        use_hard_neg: bool = True,
    ):
        super().__init__()
        self.lambda_goal = lambda_goal
        self.lambda_rel = lambda_rel
        self.lambda_geom = lambda_geom
        self.lambda_contrastive = lambda_contrastive
        self.lambda_hard_neg = lambda_hard_neg
        
        self.use_focal = use_focal
        self.use_contrastive = use_contrastive
        self.use_uncertainty = use_uncertainty
        self.use_hard_neg = use_hard_neg
        
        # Loss modules
        self.focal_loss = FocalLoss() if use_focal else None
        self.contrastive_loss = ContrastiveLoss() if use_contrastive else None
        self.periodic_angle_loss = PeriodicAngleLoss()
        self.uncertainty_loss = UncertaintyWeightedLoss() if use_uncertainty else None
        self.hard_neg_loss = HardNegativeMiningLoss() if use_hard_neg else None
        
    def compute_goal_loss(self, logits: torch.Tensor, target_index: int,
                         candidate_mask: torch.Tensor) -> torch.Tensor:
        """Compute goal selection loss."""
        # Filter to candidates
        cand_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        if cand_indices.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        logits_cand = logits[cand_indices]
        
        # Find target in candidates
        matches = torch.nonzero(cand_indices == target_index, as_tuple=False).squeeze(-1)
        if matches.numel() == 0:
            # Warn once per occurrence to help diagnose label/mask mismatch
            try:
                print(f"[Warn][GoalLoss] target_index {int(target_index)} not in candidates (num_cand={cand_indices.numel()}). Skipping this sample.")
            except Exception:
                pass
            return torch.tensor(0.0, device=logits.device)
        
        target_local = matches[0].item()
        target_tensor = torch.tensor([target_local], device=logits.device, dtype=torch.long)
        
        # Use focal loss or standard cross-entropy
        if self.use_focal:
            loss = self.focal_loss(logits_cand.unsqueeze(0), target_tensor)
        else:
            loss = F.cross_entropy(logits_cand.unsqueeze(0), target_tensor)
        
        return loss
    
    def compute_geometric_loss(self, pred_mean: torch.Tensor, pred_logvar: Optional[torch.Tensor],
                              target: torch.Tensor) -> torch.Tensor:
        """Compute geometric regression loss."""
        if self.use_uncertainty and pred_logvar is not None:
            return self.uncertainty_loss(pred_mean, pred_logvar, target)
        else:
            # Standard loss with periodic angle handling
            dist_loss = F.smooth_l1_loss(pred_mean[0], target[0])
            angle_loss = self.periodic_angle_loss(pred_mean[1], target[1])
            return dist_loss + angle_loss
    
    def forward(self, outputs: Dict[str, Any], sample: Dict[str, Any],
                device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            outputs: model outputs
            sample: data sample
            device: torch device
            
        Returns:
            dict of individual losses and total loss
        """
        goal_logits = outputs["goal_logits"]
        rel_type_logits = outputs["rel_type_logits"]
        geom_pred = outputs["geom_pred"]
        geom_logvar = outputs.get("geom_logvar", None)
        
        # Handle batching
        if goal_logits.dim() == 1:
            goal_logits = goal_logits.unsqueeze(0)
            rel_type_logits = rel_type_logits.unsqueeze(0)
            geom_pred = geom_pred.unsqueeze(0)
            if geom_logvar is not None:
                geom_logvar = geom_logvar.unsqueeze(0)
        
        B = goal_logits.size(0)
        
        losses = {
            "goal": torch.tensor(0.0, device=device),
            "rel": torch.tensor(0.0, device=device),
            "geom": torch.tensor(0.0, device=device),
            "contrastive": torch.tensor(0.0, device=device),
            "hard_neg": torch.tensor(0.0, device=device),
        }
        
        total_samples = 0
        
        for b in range(B):
            logits_b = goal_logits[b]
            rel_logits_b = rel_type_logits[b]
            geom_pred_b = geom_pred[b]
            geom_logvar_b = geom_logvar[b] if geom_logvar is not None else None
            
            # Extract sample data
            candidate_mask = sample["candidate_mask"][b] if isinstance(sample["candidate_mask"], torch.Tensor) and sample["candidate_mask"].dim() == 2 else sample["candidate_mask"]
            target_index = sample["target_index"][b].item() if isinstance(sample["target_index"], torch.Tensor) and sample["target_index"].dim() > 0 else sample["target_index"]
            rel_type = sample["rel_type"][b].item() if isinstance(sample["rel_type"], torch.Tensor) and sample["rel_type"].dim() > 0 else sample["rel_type"]
            geom_gt = sample["geom"][b] if isinstance(sample["geom"], torch.Tensor) and sample["geom"].dim() == 2 else sample["geom"]
            
            # Goal loss
            loss_goal = self.compute_goal_loss(logits_b, int(target_index), candidate_mask)
            losses["goal"] += loss_goal
            
            # Relation loss
            rel_type_tensor = torch.tensor([int(rel_type)], device=device, dtype=torch.long)
            loss_rel = F.cross_entropy(rel_logits_b.unsqueeze(0), rel_type_tensor)
            losses["rel"] += loss_rel
            
            # Geometric loss
            loss_geom = self.compute_geometric_loss(geom_pred_b, geom_logvar_b, geom_gt.to(device))
            losses["geom"] += loss_geom
            
            # Contrastive loss
            if self.use_contrastive and "node_embeddings" in outputs:
                node_emb = outputs["node_embeddings"][b] if isinstance(outputs["node_embeddings"], list) else outputs["node_embeddings"][b]
                lang_emb = outputs["lang_embedding"][b] if outputs["lang_embedding"].dim() > 1 else outputs["lang_embedding"]
                loss_contrastive = self.contrastive_loss(node_emb, lang_emb, int(target_index), candidate_mask)
                losses["contrastive"] += loss_contrastive
            
            # Hard negative mining loss
            if self.use_hard_neg:
                loss_hard_neg = self.hard_neg_loss(logits_b, int(target_index), candidate_mask)
                losses["hard_neg"] += loss_hard_neg
            
            total_samples += 1
        
        # Average over batch
        if total_samples > 0:
            for k in losses:
                losses[k] = losses[k] / total_samples
        
        # Compute total loss
        total_loss = (
            self.lambda_goal * losses["goal"] +
            self.lambda_rel * losses["rel"] +
            self.lambda_geom * losses["geom"] +
            self.lambda_contrastive * losses["contrastive"] +
            self.lambda_hard_neg * losses["hard_neg"]
        )
        
        losses["total"] = total_loss
        
        return losses

