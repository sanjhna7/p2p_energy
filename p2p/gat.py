"""Graph Attention Network encoder and actor-critic for the house agents.

The GAT runs on the scenario's real feeder subgraph (houses, junctions
and the transformer, edges = cables). Each layer lets a node attend to
its electrical neighbours, so after L layers a house "sees" every node
within L cable hops, weighted by learned attention (slide 15, block 2).

Node features (one row per graph node):
  node type        is_house, is_junction, is_transformer (one-hot)
  house state      pv, load, net power (/ battery rating), SoC
                   (zeros for junctions and the transformer)
  network state    node voltage deviation from the bus voltage after the
                   previous interval, (V - V_bus) / (V_max - V_bus)
  position         cable resistance back to the transformer (normalised)
  time             sin, cos of the hour of day

ActorCritic
  actor   house embedding -> mean of a Gaussian over [-1, 1], scaled to the
          battery's rated power. One set of weights shared by every house,
          so the same policy works for any number of houses.
  critic  house embedding + mean of all node embeddings -> value. The
          pooled term gives the critic a community-wide view during
          training (centralised training, decentralised execution).

Adapted from the HouseGAT voltage regressor (3 GATConv layers, ELU),
which ran on a synthetic tree.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

N_FEATURES = 11


class GraphContext:
    """Static graph data for one scenario, built once."""

    def __init__(self, sc, cfg):
        types = sc.nodes["node_type"].to_numpy()
        self.n_nodes = len(types)
        self.onehot = np.stack([types == "house", types == "junction", types == "substation_lv"], axis=1).astype(np.float32)
        r = sc.nodes["path_r1_ohm_from_transformer"].to_numpy(dtype=np.float32)
        self.r_norm = r / r.max() if r.max() > 0 else r
        self.house_nodes = sc.house_local_ids
        self.edge_index = torch.as_tensor(sc.edge_index, dtype=torch.long)
        self.p_scale = cfg.battery.rated_dc_power_kw.value
        self.v_bus = cfg.network.bus_voltage_v.value
        self.v_span = cfg.network.voltage_max_v.value - self.v_bus

    def features(self, obs) -> np.ndarray:
        x = np.zeros((self.n_nodes, N_FEATURES), dtype=np.float32)
        x[:, 0:3] = self.onehot
        h = self.house_nodes
        x[h, 3] = obs["pv_kw"] / self.p_scale
        x[h, 4] = obs["load_kw"] / self.p_scale
        x[h, 5] = (obs["pv_kw"] - obs["load_kw"]) / self.p_scale
        x[h, 6] = obs["soc"]
        x[:, 7] = (obs["v_node"] - self.v_bus) / self.v_span
        x[:, 8] = self.r_norm
        angle = 2 * math.pi * obs["hour"] / 24
        x[:, 9], x[:, 10] = math.sin(angle), math.cos(angle)
        return x

    def batch_edges(self, b: int) -> torch.Tensor:
        """Edge index for b copies of the graph stacked into one big graph."""
        offsets = torch.arange(b).repeat_interleave(self.edge_index.shape[1]) * self.n_nodes
        return self.edge_index.repeat(1, b) + offsets


class GATEncoder(nn.Module):
    def __init__(self, in_channels=N_FEATURES, hidden=32, heads=4, layers=3):
        super().__init__()
        dims = [in_channels] + [hidden * heads] * (layers - 1)
        self.convs = nn.ModuleList(GATConv(d, hidden, heads=heads) for d in dims)
        self.out_dim = hidden * heads

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        return x


class ActorCritic(nn.Module):
    def __init__(self, hidden=32, heads=4, layers=3):
        super().__init__()
        self.encoder = GATEncoder(hidden=hidden, heads=heads, layers=layers)
        d = self.encoder.out_dim
        self.actor = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 1))
        self.critic = nn.Sequential(nn.Linear(2 * d, 64), nn.Tanh(), nn.Linear(64, 1))
        self.log_std = nn.Parameter(torch.full((1,), -0.5))

    def forward(self, x, edge_index, n_graphs: int, n_nodes: int, house_nodes):
        """x: (n_graphs * n_nodes, F). Returns mean, std, value, each (n_graphs, n_houses)."""
        z = self.encoder(x, edge_index).view(n_graphs, n_nodes, -1)
        houses = z[:, house_nodes]                                  # (B, N, d)
        pooled = z.mean(dim=1, keepdim=True).expand_as(houses)      # (B, N, d)
        mean = torch.tanh(self.actor(houses)).squeeze(-1)
        value = self.critic(torch.cat([houses, pooled], dim=-1)).squeeze(-1)
        return mean, self.log_std.exp().expand_as(mean), value
