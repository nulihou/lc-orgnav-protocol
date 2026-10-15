import os
from typing import List, Dict, Any, Union, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset


class GoalSelectionDataset(Dataset):
    """Offline dataset for relation-aware goal selection.

    Each sample is stored in a .npz file with at least the following arrays:
        - node_feats: (N, F_node)
        - edge_index: (2, E) int64, [src, dst] indices in [0, N-1]
        - edge_attr: (E, F_edge)
        - lang_feat: (F_lang,)
        - candidate_mask: (N,) bool or {0,1}
        - target_index: () int, index in [0, N-1]
        - rel_type: () int in [0, num_rel_types-1]
        - geom: (2,) float32, [distance, theta]
        - is_online_canonical: optional bool flag for canonical online scenes

    You can extend with more fields if needed; unused keys are ignored.
    """

    def __init__(self, list_source: Union[str, Sequence[str]]):
        if isinstance(list_source, (list, tuple)):
            self.paths: List[str] = [
                str(p).strip() for p in list_source
                if str(p).strip()
            ]
        else:
            list_file = str(list_source)
            if not os.path.exists(list_file):
                raise FileNotFoundError(f"List file not found: {list_file}")
            with open(list_file, "r", encoding="utf-8") as f:
                self.paths = [
                    line.strip() for line in f.readlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
        if not self.paths:
            raise RuntimeError("No sample paths found for GoalSelectionDataset")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sample not found: {path}")
        data = np.load(path, allow_pickle=False)

        def to_tensor(key, dtype=torch.float32):
            arr = data[key]
            return torch.from_numpy(arr.astype(np.float32)) if dtype == torch.float32 else torch.from_numpy(arr.astype(np.int64))

        node_feats = to_tensor("node_feats", dtype=torch.float32)  # (N, F_node)
        edge_index = torch.from_numpy(data["edge_index"].astype(np.int64))  # (2, E)
        edge_attr = to_tensor("edge_attr", dtype=torch.float32)  # (E, F_edge)
        lang_feat = to_tensor("lang_feat", dtype=torch.float32)  # (F_lang,)
        candidate_mask = torch.from_numpy(data["candidate_mask"].astype(np.bool_))  # (N,)
        target_index = int(data["target_index"])
        rel_type = int(data["rel_type"])
        geom = to_tensor("geom", dtype=torch.float32)  # (2,)
        is_online_canonical = False
        if "is_online_canonical" in data.files:
            try:
                is_online_canonical = bool(data["is_online_canonical"].item())
            except Exception:
                is_online_canonical = bool(data["is_online_canonical"])

        sample: Dict[str, Any] = {
            "node_feats": node_feats,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "lang_feat": lang_feat,
            "candidate_mask": candidate_mask,
            "target_index": target_index,
            "rel_type": rel_type,
            "geom": geom,
            "is_online_canonical": is_online_canonical,
        }
        return sample
