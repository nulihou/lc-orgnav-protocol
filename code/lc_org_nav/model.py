
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class LCORGNet(nn.Module):
    """Language-Conditioned Object-Relation Graph Network.

    This model operates on a single scene graph:
        - node_feats: (N, F_node)
        - edge_index: (2, E)
        - edge_attr: (E, F_edge)
        - lang_feat: (F_lang,)
        - candidate_mask: (N,) bool, True for candidate vehicle nodes

    It outputs:
        - goal_logits: (N,) scores per node (we will restrict to candidates)
        - rel_type_logits: (num_rel_types,)
        - geom_pred: (2,) predicted [distance, theta] for the target
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        lang_feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_rel_types: int = 5,
        use_rel_gate: bool = False,
        use_edge_context_head: bool = False,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.lang_feat_dim = lang_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_rel_types = num_rel_types
        self.use_rel_gate = use_rel_gate
        self.use_edge_context_head = use_edge_context_head

        # initial encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.lang_proj = nn.Sequential(
            nn.Linear(lang_feat_dim, hidden_dim),
            nn.ReLU(),
        )

        # FiLM-style language-conditioned parameters for each layer
        self.film_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.film_layers.append(
                nn.Linear(hidden_dim, 4 * hidden_dim)  # gamma_node, beta_node, gamma_edge, beta_edge
            )

        # message passing weights per layer
        self.msg_mlps = nn.ModuleList()
        self.self_mlps = nn.ModuleList()
        for _ in range(num_layers):
            self.msg_mlps.append(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
            )
            self.self_mlps.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
            )

        # goal selection head
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # relation type head
        self.rel_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_rel_types),
        )

        # geometric regression head (distance, theta)
        self.geom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        # 关系 Gate 模块（轻量级，可选）
        if use_rel_gate:
            # 关系条件编码：将 lang_feat 映射到 hidden_dim
            # 注意：lang_proj 已经将 lang_feat_dim -> hidden_dim，这里可以直接复用或新建
            # 为了灵活性，我们新建一个投影层（虽然可能和 lang_proj 功能重复，但保持模块独立性）
            self.rel_proj = nn.Linear(lang_feat_dim, hidden_dim)
            
            # Gate MLP：输入拼接 [h_node, r]，输出一个 (0,1) 的 gate
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),   # 输出维度 = hidden_dim, 做逐维门控
            )

        if use_edge_context_head:
            self.raw_node_proj = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ReLU(),
            )
            self.edge_context_proj = nn.Sequential(
                nn.Linear(edge_feat_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.graph_pool_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
            )
            self.edge_goal_head = nn.Sequential(
                nn.Linear(hidden_dim * 5, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def _edge_context_from_raw_features(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Average incoming and outgoing raw edge attributes for every node."""
        device = edge_attr.device
        if edge_index.numel() == 0 or edge_attr.numel() == 0:
            return torch.zeros(num_nodes, self.edge_feat_dim * 2, device=device)

        src = edge_index[0]
        dst = edge_index[1]
        in_sum = torch.zeros(num_nodes, self.edge_feat_dim, device=device)
        out_sum = torch.zeros(num_nodes, self.edge_feat_dim, device=device)
        in_count = torch.zeros(num_nodes, 1, device=device)
        out_count = torch.zeros(num_nodes, 1, device=device)

        in_sum.index_add_(0, dst, edge_attr)
        out_sum.index_add_(0, src, edge_attr)
        ones = torch.ones(edge_attr.size(0), 1, device=device)
        in_count.index_add_(0, dst, ones)
        out_count.index_add_(0, src, ones)

        in_mean = in_sum / in_count.clamp_min(1.0)
        out_mean = out_sum / out_count.clamp_min(1.0)
        return torch.cat([in_mean, out_mean], dim=-1)

    def forward_single(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        lang_feat: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """Forward pass for a single scene.

        Args:
            node_feats: (N, F_node)
            edge_index: (2, E)
            edge_attr: (E, F_edge)
            lang_feat: (F_lang,)
            candidate_mask: (N,)

        Returns:
            dict with:
                - goal_logits: (N,)
                - rel_type_logits: (num_rel_types,)
                - geom_pred: (2,)
        """
        device = node_feats.device
        
        # Helper function to squeeze singleton batch dimension
        def _squeeze_if_singleton(x, name):
            if x is None:
                return None
            if x.dim() == 3 and x.size(0) == 1:
                return x.squeeze(0)
            return x
        
        # Standardize inputs: remove singleton batch dimensions
        node_feats = _squeeze_if_singleton(node_feats, "node_feats")   # [N, F]
        edge_attr = _squeeze_if_singleton(edge_attr, "edge_attr")      # [E, F_e]
        edge_index = _squeeze_if_singleton(edge_index, "edge_index")    # [2, E]
        
        # Assertions to catch dimension errors early
        assert node_feats.dim() == 2, f"node_feats must be [N,F], got {node_feats.shape}"
        assert edge_index.dim() == 2 and edge_index.size(0) == 2, f"edge_index must be [2,E], got {edge_index.shape}"
        if edge_attr is not None:
            assert edge_attr.dim() == 2, f"edge_attr must be [E,F_e], got {edge_attr.shape}"
            assert edge_attr.size(0) == edge_index.size(1), f"edge_attr rows != E: {edge_attr.size(0)} vs {edge_index.size(1)}"
        
        N = node_feats.size(0)
        E = edge_index.size(1)

        h_nodes = self.node_encoder(node_feats)        # (N, H)
        h_edges = self.edge_encoder(edge_attr)         # (E, H)
        h_lang = self.lang_proj(lang_feat)             # (H,)

        # message passing
        for layer_idx in range(self.num_layers):
            film = self.film_layers[layer_idx](h_lang)  # (4H,)
            gamma_node, beta_node, gamma_edge, beta_edge = torch.chunk(film, 4, dim=-1)

            # apply FiLM to node and edge features
            h_nodes_f = h_nodes * gamma_node + beta_node           # (N, H)
            h_edges_f = h_edges * gamma_edge + beta_edge           # (E, H)

            src = edge_index[0]    # (E,)
            dst = edge_index[1]    # (E,)

            # build messages per edge
            h_src = h_nodes_f[src]                     # (E, H)
            msg_input = torch.cat([h_src, h_edges_f], dim=-1)  # (E, 2H)
            msgs = self.msg_mlps[layer_idx](msg_input)         # (E, H)

            # aggregate messages to dst nodes
            agg = torch.zeros(N, self.hidden_dim, device=device)
            agg.index_add_(0, dst, msgs)               # sum messages per dst

            # self update
            self_update = self.self_mlps[layer_idx](h_nodes_f)   # (N, H)

            h_nodes = F.relu(self_update + agg)        # (N, H)

        # 关系 Gate（轻量级，在 readout/goal selection 之前）
        if self.use_rel_gate:
            # lang_feat: (F_lang,)，通过 rel_proj 映射到 hidden_dim
            r = self.rel_proj(lang_feat)              # (H,)
            
            # 扩展到每个节点
            r_exp = r.unsqueeze(0).expand(N, -1)      # (N, H)
            
            # 拼接 [h_nodes, r_exp] → (N, 2H)
            gate_in = torch.cat([h_nodes, r_exp], dim=-1)  # (N, 2H)
            gate = self.gate_mlp(gate_in)             # (N, H) in (0,1)
            
            # gate 融合：关系条件自适应调节节点表示
            h_nodes = gate * h_nodes + (1.0 - gate) * r_exp  # (N, H)

        # goal selection: compute score for each node, conditioned on language
        lang_expanded = h_lang.unsqueeze(0).expand(N, -1)   # (N, H)
        goal_input = torch.cat([h_nodes, lang_expanded], dim=-1)  # (N, 2H)
        goal_logits = self.goal_head(goal_input).squeeze(-1)      # (N,)

        if self.use_edge_context_head:
            edge_context = self._edge_context_from_raw_features(edge_index, edge_attr, N)
            raw_h = self.raw_node_proj(node_feats)
            edge_h = self.edge_context_proj(edge_context)
            mean_pool = torch.mean(h_nodes, dim=0)
            max_pool = torch.max(h_nodes, dim=0)[0]
            graph_h = self.graph_pool_proj(torch.cat([mean_pool, max_pool], dim=-1))
            graph_expanded = graph_h.unsqueeze(0).expand(N, -1)
            edge_goal_input = torch.cat(
                [h_nodes, lang_expanded, raw_h, edge_h, graph_expanded],
                dim=-1,
            )
            goal_logits = goal_logits + self.edge_goal_head(edge_goal_input).squeeze(-1)

        # relation type head: use pooled anchor+target-like representation
        # Here we simply use global max pooling as a simple proxy; during training
        # we still supervise with anchor-target label.
        pooled = torch.max(h_nodes, dim=0)[0]                  # (H,)
        rel_input = torch.cat([pooled, h_lang], dim=-1)        # (2H,)
        rel_type_logits = self.rel_head(rel_input)             # (num_rel_types,)

        # geometric regression head: use a weighted sum over candidate nodes
        cand_mask = candidate_mask.to(torch.float32)           # (N,)
        if cand_mask.sum() > 0:
            weights = cand_mask / (cand_mask.sum() + 1e-6)
            geom_repr = (weights.unsqueeze(-1) * h_nodes).sum(dim=0)  # (H,)
        else:
            geom_repr = pooled
        geom_pred = self.geom_head(geom_repr)                  # (2,)

        return {
            "goal_logits": goal_logits,
            "rel_type_logits": rel_type_logits,
            "geom_pred": geom_pred,
        }

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        lang_feat: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """Batched forward.

        We keep batch_size=1 in the trainer for simplicity, but this method
        supports batch_size >= 1 by iterating over scenes.
        """
        if isinstance(node_feats, list):
            B = len(node_feats)
            outputs: Dict[str, Any] = {
                "goal_logits": [],
                "rel_type_logits": [],
                "geom_pred": [],
            }
            for b in range(B):
                out_b = self.forward_single(
                    node_feats[b],
                    edge_index[b],
                    edge_attr[b],
                    lang_feat[b],
                    candidate_mask[b],
                )
                outputs["goal_logits"].append(out_b["goal_logits"])
                outputs["rel_type_logits"].append(out_b["rel_type_logits"])
                outputs["geom_pred"].append(out_b["geom_pred"])

            max_n = max(item.size(0) for item in outputs["goal_logits"])
            padded_logits = []
            for logits in outputs["goal_logits"]:
                if logits.size(0) < max_n:
                    pad = torch.full((max_n - logits.size(0),), float("-inf"), device=logits.device)
                    padded_logits.append(torch.cat([logits, pad], dim=0))
                else:
                    padded_logits.append(logits)

            outputs["goal_logits"] = torch.stack(padded_logits, dim=0)
            outputs["rel_type_logits"] = torch.stack(outputs["rel_type_logits"], dim=0)
            outputs["geom_pred"] = torch.stack(outputs["geom_pred"], dim=0)
            return outputs

        if node_feats.dim() == 2:
            # single scene
            return self.forward_single(node_feats, edge_index, edge_attr, lang_feat, candidate_mask)

        B = node_feats.size(0)
        outputs: Dict[str, Any] = {
            "goal_logits": [],
            "rel_type_logits": [],
            "geom_pred": [],
        }
        for b in range(B):
            out_b = self.forward_single(
                node_feats[b],
                edge_index[b],
                edge_attr[b],
                lang_feat[b],
                candidate_mask[b],
            )
            outputs["goal_logits"].append(out_b["goal_logits"])
            outputs["rel_type_logits"].append(out_b["rel_type_logits"])
            outputs["geom_pred"].append(out_b["geom_pred"])

        outputs["goal_logits"] = torch.stack(outputs["goal_logits"], dim=0)
        outputs["rel_type_logits"] = torch.stack(outputs["rel_type_logits"], dim=0)
        outputs["geom_pred"] = torch.stack(outputs["geom_pred"], dim=0)
        return outputs
