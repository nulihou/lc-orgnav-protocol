"""
Rule-based baseline for goal selection.

Uses geometric heuristics and language parsing to select target vehicle.
"""

import math
from typing import Dict, Any, List
import torch
import numpy as np


class RuleBasedSelector:
    """Rule-based goal selector using geometric heuristics."""
    
    def __init__(self, num_rel_types: int = 5):
        self.num_rel_types = num_rel_types
        self._device = torch.device("cpu")
    
    def to(self, device):
        """Compatibility method for nn.Module interface."""
        self._device = device
        return self
    
    def eval(self):
        """Compatibility method for nn.Module interface."""
        return self
    
    @property
    def relation_rules(self):
        """Relation type to geometric preference mapping."""
        return {
            0: self._select_nearest,      # near
            1: self._select_front,         # front
            2: self._select_behind,        # behind
            3: self._select_left,          # left
            4: self._select_right,         # right
        }
    
    def _extract_geometry_from_features(self, node_feats: torch.Tensor) -> Dict[int, Dict]:
        """Extract geometric information from node features."""
        # Handle batch dimension if present
        if node_feats.dim() == 3:
            node_feats = node_feats.squeeze(0)
        
        # Assuming node features contain: [dx, dy, distance, cos_theta, sin_theta, ...]
        geometries = {}
        
        for i in range(node_feats.size(0)):
            feat = node_feats[i]
            dx = feat[0].item() if feat[0].numel() == 1 else feat[0].cpu().numpy()[0]
            dy = feat[1].item() if feat[1].numel() == 1 else feat[1].cpu().numpy()[0]
            distance = feat[2].item() if feat[2].numel() == 1 else feat[2].cpu().numpy()[0]
            cos_theta = feat[3].item() if feat[3].numel() == 1 else feat[3].cpu().numpy()[0]
            sin_theta = feat[4].item() if feat[4].numel() == 1 else feat[4].cpu().numpy()[0]
            theta = math.atan2(sin_theta, cos_theta)
            
            geometries[i] = {
                "dx": dx,
                "dy": dy,
                "distance": distance,
                "theta": theta,
                "angle_deg": math.degrees(theta),
            }
        
        return geometries
    
    def _select_nearest(self, geometries: Dict, candidates: List[int]) -> int:
        """Select nearest vehicle."""
        min_dist = float('inf')
        best_idx = candidates[0]
        
        for idx in candidates:
            if geometries[idx]["distance"] < min_dist:
                min_dist = geometries[idx]["distance"]
                best_idx = idx
        
        return best_idx
    
    def _select_front(self, geometries: Dict, candidates: List[int]) -> int:
        """Select vehicle in front (smallest angle, positive distance)."""
        best_score = float('inf')
        best_idx = candidates[0]
        
        for idx in candidates:
            angle = abs(geometries[idx]["angle_deg"])
            distance = geometries[idx]["distance"]
            
            # Prefer small angle and reasonable distance
            if -45 <= geometries[idx]["angle_deg"] <= 45:
                score = angle + distance / 10.0  # Weighted score
                if score < best_score:
                    best_score = score
                    best_idx = idx
        
        return best_idx
    
    def _select_behind(self, geometries: Dict, candidates: List[int]) -> int:
        """Select vehicle behind."""
        best_score = float('inf')
        best_idx = candidates[0]
        
        for idx in candidates:
            angle_deg = geometries[idx]["angle_deg"]
            distance = geometries[idx]["distance"]
            
            # Behind: angle near ±180°
            if abs(angle_deg) > 135:
                score = abs(abs(angle_deg) - 180) + distance / 10.0
                if score < best_score:
                    best_score = score
                    best_idx = idx
        
        return best_idx
    
    def _select_left(self, geometries: Dict, candidates: List[int]) -> int:
        """Select vehicle on left side."""
        best_score = float('inf')
        best_idx = candidates[0]
        
        for idx in candidates:
            angle_deg = geometries[idx]["angle_deg"]
            distance = geometries[idx]["distance"]
            
            # Left: angle 45-135°
            if 45 < angle_deg <= 135:
                score = abs(angle_deg - 90) + distance / 10.0
                if score < best_score:
                    best_score = score
                    best_idx = idx
        
        return best_idx
    
    def _select_right(self, geometries: Dict, candidates: List[int]) -> int:
        """Select vehicle on right side."""
        best_score = float('inf')
        best_idx = candidates[0]
        
        for idx in candidates:
            angle_deg = geometries[idx]["angle_deg"]
            distance = geometries[idx]["distance"]
            
            # Right: angle -135 to -45°
            if -135 < angle_deg <= -45:
                score = abs(angle_deg + 90) + distance / 10.0
                if score < best_score:
                    best_score = score
                    best_idx = idx
        
        return best_idx
    
    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        lang_feat: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Forward pass (mimics neural network interface).
        
        Args:
            node_feats: (N, F_node)
            edge_index: (2, E) - not used
            edge_attr: (E, F_edge) - not used
            lang_feat: (F_lang,)
            candidate_mask: (N,)
            
        Returns:
            dict with goal_logits, rel_type_logits, geom_pred
        """
        N = node_feats.size(0)
        device = node_feats.device
        
        # Handle batch dimension
        if candidate_mask.dim() > 1:
            candidate_mask = candidate_mask.squeeze(0)
        if lang_feat.dim() > 1:
            lang_feat = lang_feat.squeeze(0)
        
        # Parse relation type from language feature (one-hot in first 5 dims)
        rel_type = torch.argmax(lang_feat[:self.num_rel_types]).item()
        
        # Extract geometries
        geometries = self._extract_geometry_from_features(node_feats)
        
        # Get candidate indices - ensure it's a flat list of ints
        cand_indices = []
        for i in range(N):
            if candidate_mask[i].item():
                cand_indices.append(i)
        
        if len(cand_indices) == 0:
            # No candidates, return dummy
            return {
                "goal_logits": torch.zeros(N, device=device),
                "rel_type_logits": torch.zeros(self.num_rel_types, device=device),
                "geom_pred": torch.zeros(2, device=device),
            }
        
        # Apply rule
        if rel_type in self.relation_rules:
            selected_idx = self.relation_rules[rel_type](geometries, cand_indices)
        else:
            selected_idx = self._select_nearest(geometries, cand_indices)
        
        # Create goal logits (one-hot style)
        goal_logits = torch.zeros(N, device=device)
        goal_logits[selected_idx] = 10.0  # High score for selected
        
        # Relation type logits (one-hot)
        rel_type_logits = torch.zeros(self.num_rel_types, device=device)
        rel_type_logits[rel_type] = 10.0
        
        # Geometric prediction (use selected vehicle's geometry)
        geom_pred = torch.tensor([
            geometries[selected_idx]["distance"],
            geometries[selected_idx]["theta"],
        ], device=device)
        
        return {
            "goal_logits": goal_logits,
            "rel_type_logits": rel_type_logits,
            "geom_pred": geom_pred,
        }
    
    def __call__(self, *args, **kwargs):
        """Make it callable like a nn.Module."""
        return self.forward(*args, **kwargs)

