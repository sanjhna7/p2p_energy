"""
Graph Attention Network (GAT) for house-level voltage prediction.

This module implements a GAT that predicts per-house voltage magnitudes
on a synthetic Low-Voltage (LV) distribution network. Unlike IEEE-33
bus-level models, each node represents ONE individual house's service
connection point, and node 0 is the distribution transformer.

Components:
    - build_lv_tree()            — synthetic radial LV network topology
    - generate_synthetic_house_data() — per-house load + voltage data
    - HouseGAT                   — 3-layer GAT regression model
    - run_gat()                  — full training/evaluation pipeline
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# =====================================================================
# 1. Synthetic LV network topology
# =====================================================================

def build_lv_tree(num_houses=30, num_laterals=4, seed=42):
    """Build a radial tree topology mimicking a real LV feeder.

    A few main laterals branch off the transformer (node 0), each with
    houses strung along it and occasional short side-taps.
    """
    rng = np.random.default_rng(seed)
    branches = []
    node_id = 1  # 0 is reserved for the transformer

    for lateral in range(num_laterals):
        prev = 0
        lateral_len = num_houses // num_laterals
        for i in range(lateral_len):
            branches.append((prev, node_id))
            if i > 0 and rng.random() < 0.15 and node_id + 1 <= num_houses:
                side_parent = node_id
                node_id += 1
                branches.append((side_parent, node_id))
            prev = node_id
            node_id += 1
            if node_id > num_houses:
                break

    return branches


def _count_nodes(branches):
    """Count actual nodes built (may differ from num_houses due to side-taps)."""
    built_nodes = {0}
    for a, b in branches:
        built_nodes.add(a)
        built_nodes.add(b)
    return len(built_nodes)


def compute_depth_from_root(branches, N, root=0):
    """BFS distance (# hops) from the transformer to every house."""
    from collections import deque, defaultdict
    adj = defaultdict(list)
    for a, b in branches:
        adj[a].append(b)
        adj[b].append(a)
    depth = np.zeros(N, dtype=np.float32)
    visited = {root}
    q = deque([root])
    while q:
        node = q.popleft()
        for nbr in adj[node]:
            if nbr not in visited:
                visited.add(nbr)
                depth[nbr] = depth[node] + 1
                q.append(nbr)
    return depth


# =====================================================================
# 2. Synthetic per-house data generation
# =====================================================================

def generate_synthetic_house_data(num_samples=500, N=31, branches=None, seed=0):
    """Generate synthetic per-house load features and voltage labels.

    Features per node: [P (active power), Q (reactive power),
                        has_solar (binary), has_ev (binary)]
    Labels: voltage magnitude at each node.
    """
    rng = np.random.default_rng(seed)

    # Fixed per-house attributes
    has_solar = (rng.random(N) < 0.3).astype(np.float32)
    has_ev = (rng.random(N) < 0.2).astype(np.float32)
    has_solar[0], has_ev[0] = 0.0, 0.0  # transformer node

    if branches is None:
        branches = build_lv_tree(num_houses=N - 1)

    X_list, Y_list = [], []
    for _ in range(num_samples):
        P = rng.uniform(0.01, 0.06, size=N).astype(np.float32)
        Q = rng.uniform(0.005, 0.02, size=N).astype(np.float32)

        solar_gen = has_solar * rng.uniform(0.0, 0.08, size=N).astype(np.float32)
        P = P - solar_gen

        ev_active = has_ev * (rng.random(N) < 0.4)
        P = P + ev_active * rng.uniform(0.03, 0.09, size=N).astype(np.float32)

        P[0], Q[0] = 0.0, 0.0  # transformer has no injection

        depth = compute_depth_from_root(branches, N)
        voltage = 1.0 - 0.01 * depth * (0.5 + np.abs(P)) - 0.005 * depth * np.abs(Q)
        voltage += rng.normal(0, 0.001, size=N)

        features = np.stack([P, Q, has_solar, has_ev], axis=1)
        X_list.append(features)
        Y_list.append(voltage.astype(np.float32))

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


# =====================================================================
# 3. GAT model
# =====================================================================

class HouseGAT(nn.Module):
    """3-layer GAT for per-node voltage regression."""

    def __init__(self, in_channels=4, hidden_channels=32, out_channels=1, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.2)
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        return x.squeeze(-1)   # (N,) predicted voltage per house


# =====================================================================
# 4. Training / evaluation
# =====================================================================

def run_gat(num_houses=30, num_laterals=4, num_samples=500, epochs=50):
    """Full GAT training pipeline.

    Builds the LV tree, generates synthetic data, trains the GAT, and
    evaluates on a held-out test set.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Build topology
    branches = build_lv_tree(num_houses=num_houses, num_laterals=num_laterals, seed=42)
    N = _count_nodes(branches)

    edge_list = branches + [(b, a) for (a, b) in branches]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    print(f"Total nodes: {N} (1 transformer + {N - 1} houses), "
          f"Branches: {len(branches)}")

    # Generate data
    X_all, Y_all = generate_synthetic_house_data(
        num_samples=num_samples, N=N, branches=branches
    )

    # Train/val/test split
    n_train, n_val = 350, 75
    X_train, Y_train = X_all[:n_train], Y_all[:n_train]
    X_val, Y_val = X_all[n_train:n_train + n_val], Y_all[n_train:n_train + n_val]
    X_test, Y_test = X_all[n_train + n_val:], Y_all[n_train + n_val:]

    print(f"Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HouseGAT(in_channels=X_all.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    edge_index = edge_index.to(device)

    def run_epoch(X, Y, train=True):
        model.train() if train else model.eval()
        total_loss = 0.0
        indices = np.random.permutation(len(X)) if train else range(len(X))

        for i in indices:
            x = torch.tensor(X[i]).to(device)
            y = torch.tensor(Y[i]).to(device)

            if train:
                optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.mse_loss(out, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(X)

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(X_train, Y_train, train=True)
        if epoch % 5 == 0:
            with torch.no_grad():
                val_loss = run_epoch(X_val, Y_val, train=False)
            print(f"Epoch {epoch:02d} | Train MSE: {train_loss:.6f} "
                  f"| Val MSE: {val_loss:.6f}")

    # Test evaluation
    with torch.no_grad():
        test_loss = run_epoch(X_test, Y_test, train=False)
    print(f"\nFinal Test MSE: {test_loss:.6f}")

    # Sanity check
    with torch.no_grad():
        x_sample = torch.tensor(X_test[0]).to(device)
        y_sample = Y_test[0]
        pred = model(x_sample, edge_index).cpu().numpy()
        print("\nSample house voltages (transformer + first 5 houses):")
        print(f"Predicted: {pred[:6]}")
        print(f"Actual:    {y_sample[:6]}")

    return model, test_loss


if __name__ == "__main__":
    run_gat()
