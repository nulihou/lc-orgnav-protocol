"""
Inference utilities for LC-ORGNav.

Includes:
1. Single sample inference
2. Batch inference
3. Model export (ONNX, TorchScript)
4. Inference optimization
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import time

import torch
import torch.nn as nn
import numpy as np

from .model_advanced import LCORGNetAdvanced
from .model import LCORGNet


class InferenceEngine:
    """Optimized inference engine for LC-ORGNav."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        use_amp: bool = True,
        use_ema: bool = False,
        ema_checkpoint: Optional[Dict] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.use_amp = use_amp and self.device.type == 'cuda'
        
        # Load EMA weights if available
        if use_ema and ema_checkpoint is not None and "ema_shadow" in ema_checkpoint:
            self._load_ema_weights(ema_checkpoint["ema_shadow"])
        
        # Warmup for accurate timing
        self._warmup()
    
    def _load_ema_weights(self, ema_shadow: Dict[str, torch.Tensor]):
        """Load EMA shadow weights into model."""
        model_dict = self.model.state_dict()
        for name, param in ema_shadow.items():
            if name in model_dict:
                model_dict[name] = param
        self.model.load_state_dict(model_dict)
        print("[Inference] Loaded EMA weights")
    
    def _warmup(self, num_iterations: int = 10):
        """Warmup the model for stable timing."""
        print("[Inference] Warming up...")
        dummy_sample = {
            "node_feats": torch.randn(10, 32, device=self.device),
            "edge_index": torch.randint(0, 10, (2, 50), device=self.device),
            "edge_attr": torch.randn(50, 8, device=self.device),
            "lang_feat": torch.randn(32, device=self.device),
            "candidate_mask": torch.ones(10, dtype=torch.bool, device=self.device),
        }
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(**dummy_sample, return_embeddings=False)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        print("[Inference] Warmup complete")
    
    @torch.no_grad()
    def predict(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        lang_feat: torch.Tensor,
        candidate_mask: torch.Tensor,
        return_all: bool = False,
    ) -> Dict[str, Any]:
        """Run inference on a single sample.
        
        Args:
            node_feats: (N, F_node) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, F_edge) edge attributes
            lang_feat: (F_lang,) language features
            candidate_mask: (N,) candidate mask
            return_all: whether to return all outputs or just predictions
            
        Returns:
            Dictionary with predictions
        """
        # Move to device
        node_feats = node_feats.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        lang_feat = lang_feat.to(self.device)
        candidate_mask = candidate_mask.to(self.device)
        
        # Inference with mixed precision
        start_time = time.time()
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(
                node_feats=node_feats,
                edge_index=edge_index,
                edge_attr=edge_attr,
                lang_feat=lang_feat,
                candidate_mask=candidate_mask,
                return_embeddings=False,
            )
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        inference_time = time.time() - start_time
        
        # Extract predictions
        goal_logits = outputs["goal_logits"]
        rel_type_logits = outputs["rel_type_logits"]
        geom_pred = outputs["geom_pred"]

        # Squeeze potential batch dimension coming from DataLoader (B=1)
        if isinstance(goal_logits, torch.Tensor) and goal_logits.dim() == 2 and goal_logits.size(0) == 1:
            goal_logits = goal_logits[0]
        if isinstance(rel_type_logits, torch.Tensor) and rel_type_logits.dim() == 2 and rel_type_logits.size(0) == 1:
            rel_type_logits = rel_type_logits[0]
        if isinstance(geom_pred, torch.Tensor) and geom_pred.dim() == 2 and geom_pred.size(0) == 1:
            geom_pred = geom_pred[0]
        if isinstance(candidate_mask, torch.Tensor) and candidate_mask.dim() == 2 and candidate_mask.size(0) == 1:
            candidate_mask = candidate_mask[0]
        candidate_mask = candidate_mask.to(torch.bool)
        
        # Get candidate indices
        cand_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        
        if cand_indices.numel() == 0:
            # No candidates
            predictions = {
                "target_index": -1,
                "target_score": 0.0,
                "rel_type": 0,
                "rel_confidence": 0.0,
                "distance": 0.0,
                "angle": 0.0,
                "inference_time": inference_time,
            }
        else:
            # Goal selection
            # Guard: ensure indices are within goal_logits length
            num_nodes = goal_logits.size(0)
            valid_mask = (cand_indices >= 0) & (cand_indices < num_nodes)
            cand_indices = cand_indices[valid_mask]
            if cand_indices.numel() == 0:
                predictions = {
                    "target_index": -1,
                    "target_score": 0.0,
                    "rel_type": int(torch.argmax(torch.softmax(rel_type_logits, dim=0)).item()),
                    "rel_confidence": float(torch.max(torch.softmax(rel_type_logits, dim=0)).item()),
                    "distance": float(geom_pred[0].item()),
                    "angle": float(geom_pred[1].item()),
                    "inference_time": inference_time,
                }
                if "geom_logvar" in outputs:
                    geom_logvar = outputs["geom_logvar"]
                    if geom_logvar.dim() == 2 and geom_logvar.size(0) == 1:
                        geom_logvar = geom_logvar[0]
                    predictions["distance_uncertainty"] = float(torch.exp(geom_logvar[0]).item())
                    predictions["angle_uncertainty"] = float(torch.exp(geom_logvar[1]).item())
                if return_all:
                    predictions["raw_outputs"] = {
                        "goal_logits": goal_logits.cpu(),
                        "rel_type_logits": rel_type_logits.cpu(),
                        "geom_pred": geom_pred.cpu(),
                    }
                return predictions

            logits_cand = goal_logits[cand_indices]
            scores_cand = torch.softmax(logits_cand, dim=0)
            pred_local = torch.argmax(scores_cand).item()
            pred_global = cand_indices[pred_local].item()
            pred_score = scores_cand[pred_local].item()
            
            # Relation type
            rel_probs = torch.softmax(rel_type_logits, dim=0)
            pred_rel_type = torch.argmax(rel_probs).item()
            pred_rel_conf = rel_probs[pred_rel_type].item()
            
            # Geometry
            pred_distance = geom_pred[0].item()
            pred_angle = geom_pred[1].item()
            
            predictions = {
                "target_index": pred_global,
                "target_score": pred_score,
                "top_k_indices": cand_indices[torch.topk(scores_cand, min(5, len(scores_cand)))[1]].cpu().tolist(),
                "top_k_scores": torch.topk(scores_cand, min(5, len(scores_cand)))[0].cpu().tolist(),
                "rel_type": pred_rel_type,
                "rel_confidence": pred_rel_conf,
                "rel_probs": rel_probs.cpu().tolist(),
                "distance": pred_distance,
                "angle": pred_angle,
                "inference_time": inference_time,
            }
            
            if "geom_logvar" in outputs:
                geom_logvar = outputs["geom_logvar"]
                # Normalize potential batch dims to shape (..., 2)
                if isinstance(geom_logvar, torch.Tensor):
                    if geom_logvar.dim() == 2 and geom_logvar.size(0) == 1:
                        geom_logvar = geom_logvar[0]
                    # If still 2D, prefer to index last dimension
                    if geom_logvar.dim() == 2 and geom_logvar.size(-1) == 2:
                        dist_logvar = geom_logvar[..., 0]
                        angle_logvar = geom_logvar[..., 1]
                        # Reduce to scalar if needed
                        if dist_logvar.numel() > 1:
                            dist_logvar = dist_logvar.view(-1)[0]
                        if angle_logvar.numel() > 1:
                            angle_logvar = angle_logvar.view(-1)[0]
                        predictions["distance_uncertainty"] = float(torch.exp(dist_logvar).item())
                        predictions["angle_uncertainty"] = float(torch.exp(angle_logvar).item())
                    elif geom_logvar.dim() == 1 and geom_logvar.numel() >= 2:
                        predictions["distance_uncertainty"] = float(torch.exp(geom_logvar[0]).item())
                        predictions["angle_uncertainty"] = float(torch.exp(geom_logvar[1]).item())
        
        if return_all:
            predictions["raw_outputs"] = {
                "goal_logits": goal_logits.cpu(),
                "rel_type_logits": rel_type_logits.cpu(),
                "geom_pred": geom_pred.cpu(),
            }
        
        return predictions
    
    @torch.no_grad()
    def predict_batch(
        self,
        samples: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, Any]]:
        """Run inference on a batch of samples.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for sample in samples:
            pred = self.predict(
                node_feats=sample["node_feats"],
                edge_index=sample["edge_index"],
                edge_attr=sample["edge_attr"],
                lang_feat=sample["lang_feat"],
                candidate_mask=sample["candidate_mask"],
                return_all=False,
            )
            predictions.append(pred)
        
        return predictions
    
    def benchmark(self, num_samples: int = 100) -> Dict[str, float]:
        """Benchmark inference speed.
        
        Args:
            num_samples: number of samples to test
            
        Returns:
            Benchmark statistics
        """
        print(f"[Benchmark] Running {num_samples} inferences...")
        
        times = []
        
        for _ in range(num_samples):
            # Generate random sample
            num_nodes = np.random.randint(8, 20)
            num_edges = num_nodes * (num_nodes - 1)
            
            sample = {
                "node_feats": torch.randn(num_nodes, 32, device=self.device),
                "edge_index": torch.randint(0, num_nodes, (2, num_edges), device=self.device),
                "edge_attr": torch.randn(num_edges, 8, device=self.device),
                "lang_feat": torch.randn(32, device=self.device),
                "candidate_mask": torch.rand(num_nodes, device=self.device) > 0.5,
            }
            
            pred = self.predict(**sample, return_all=False)
            times.append(pred["inference_time"])
        
        stats = {
            "mean_time": np.mean(times) * 1000,  # ms
            "std_time": np.std(times) * 1000,
            "min_time": np.min(times) * 1000,
            "max_time": np.max(times) * 1000,
            "median_time": np.median(times) * 1000,
            "throughput": 1.0 / np.mean(times),  # samples/sec
        }
        
        print(f"[Benchmark] Results:")
        print(f"  Mean: {stats['mean_time']:.2f} ms")
        print(f"  Std: {stats['std_time']:.2f} ms")
        print(f"  Median: {stats['median_time']:.2f} ms")
        print(f"  Throughput: {stats['throughput']:.2f} samples/sec")
        
        return stats


def load_model_for_inference(
    checkpoint_path: str,
    device: str = "cuda",
    use_amp: bool = True,
    use_ema: bool = True,
) -> InferenceEngine:
    """Load a trained model for inference.
    
    Args:
        checkpoint_path: path to checkpoint file
        device: device to run on
        use_amp: use automatic mixed precision
        use_ema: use EMA weights if available
        
    Returns:
        InferenceEngine instance
    """
    print(f"[Inference] Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Get config
    cfg = checkpoint.get("config")
    if cfg is None:
        raise ValueError("Checkpoint does not contain config")
    
    # Determine model type
    if hasattr(cfg, 'num_heads'):
        # Advanced model
        model = LCORGNetAdvanced(
            node_feat_dim=cfg.node_feat_dim,
            edge_feat_dim=cfg.edge_feat_dim,
            lang_feat_dim=cfg.lang_feat_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_rel_types=cfg.num_rel_types,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
    else:
        # Basic model
        model = LCORGNet(
            node_feat_dim=cfg.node_feat_dim,
            edge_feat_dim=cfg.edge_feat_dim,
            lang_feat_dim=cfg.lang_feat_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_rel_types=cfg.num_rel_types,
        )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Inference] Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create inference engine
    engine = InferenceEngine(
        model=model,
        device=device,
        use_amp=use_amp,
        use_ema=use_ema,
        ema_checkpoint=checkpoint if use_ema else None,
    )
    
    return engine


def export_to_torchscript(
    model: nn.Module,
    save_path: str,
    example_inputs: Optional[Dict[str, torch.Tensor]] = None,
):
    """Export model to TorchScript format.
    
    Args:
        model: PyTorch model
        save_path: path to save TorchScript model
        example_inputs: example inputs for tracing
    """
    model.eval()
    
    if example_inputs is None:
        # Create dummy inputs
        example_inputs = {
            "node_feats": torch.randn(10, 32),
            "edge_index": torch.randint(0, 10, (2, 50)),
            "edge_attr": torch.randn(50, 8),
            "lang_feat": torch.randn(32),
            "candidate_mask": torch.ones(10, dtype=torch.bool),
        }
    
    print("[Export] Tracing model with TorchScript...")
    
    try:
        traced_model = torch.jit.trace(
            model,
            example_kwarg_inputs=example_inputs,
        )
        
        torch.jit.save(traced_model, save_path)
        print(f"[Export] TorchScript model saved to {save_path}")
        
    except Exception as e:
        print(f"[Export] TorchScript export failed: {e}")
        print("[Export] Trying scripting instead of tracing...")
        
        try:
            scripted_model = torch.jit.script(model)
            torch.jit.save(scripted_model, save_path)
            print(f"[Export] TorchScript model (scripted) saved to {save_path}")
        except Exception as e2:
            print(f"[Export] Scripting also failed: {e2}")


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    example_inputs: Optional[Dict[str, torch.Tensor]] = None,
    opset_version: int = 14,
):
    """Export model to ONNX format.
    
    Args:
        model: PyTorch model
        save_path: path to save ONNX model
        example_inputs: example inputs for export
        opset_version: ONNX opset version
    """
    model.eval()
    
    if example_inputs is None:
        example_inputs = {
            "node_feats": torch.randn(10, 32),
            "edge_index": torch.randint(0, 10, (2, 50)),
            "edge_attr": torch.randn(50, 8),
            "lang_feat": torch.randn(32),
            "candidate_mask": torch.ones(10, dtype=torch.bool),
        }
    
    print("[Export] Exporting to ONNX...")
    
    try:
        torch.onnx.export(
            model,
            tuple(example_inputs.values()),
            save_path,
            input_names=list(example_inputs.keys()),
            output_names=["goal_logits", "rel_type_logits", "geom_pred"],
            dynamic_axes={
                "node_feats": {0: "num_nodes"},
                "edge_index": {1: "num_edges"},
                "edge_attr": {0: "num_edges"},
                "candidate_mask": {0: "num_nodes"},
            },
            opset_version=opset_version,
        )
        print(f"[Export] ONNX model saved to {save_path}")
        
    except Exception as e:
        print(f"[Export] ONNX export failed: {e}")

