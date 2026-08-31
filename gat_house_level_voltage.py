import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------
# 1. Synthetic LV (Low-Voltage) Network Topology
# ---------------------------------------------------------
# Unlike IEEE-33 (where each bus = an AGGREGATE of many customers),
# here node 0 = the distribution transformer (secondary side / root),
# and every other node = ONE individual house's service connection point.
#
# Real LV networks are radial trees fanning out from a transformer,
# typically feeding 20-100 houses over a few branching "laterals"
# (streets). We build a small synthetic tree in that shape: a few
# main laterals off the transformer, each with houses strung along it
# and occasional short branches (like houses on a side street/cul-de-sac).

def build_lv_tree(num_houses=30, num_laterals=4, seed=42):
    rng = np.random.default_rng(seed)
    branches = []
    node_id = 1  # 0 is reserved for the transformer

    lateral_roots = []
    for lateral in range(num_laterals):
        # Each lateral starts directly off the transformer (node 0)
        prev = 0
        lateral_len = num_houses // num_laterals
        for i in range(lateral_len):
            branches.append((prev, node_id))
            # Occasionally branch off a short side-tap (a cul-de-sac / side street)
            if i > 0 and rng.random() < 0.15 and node_id + 1 <= num_houses:
                side_parent = node_id
                node_id += 1
                branches.append((side_parent, node_id))
            prev = node_id
            node_id += 1
            if node_id > num_houses:
                break
        lateral_roots.append(lateral)

    return branches

NUM_HOUSES = 30          # houses (excludes the transformer node)
N = NUM_HOUSES + 1       # total nodes = transformer + houses, node 0 = transformer

branches = build_lv_tree(num_houses=NUM_HOUSES, num_laterals=4, seed=42)
# Some branch-building can undershoot NUM_HOUSES slightly depending on random
# side-taps; trim node count to whatever was actually built.
built_nodes = set([0])
for a, b in branches:
    built_nodes.add(a)
    built_nodes.add(b)
N = len(built_nodes)

edge_list = branches + [(b, a) for (a, b) in branches]
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

print(f"Total nodes: {N} (1 transformer + {N - 1} houses), Branches: {len(branches)}")

# ---------------------------------------------------------
# 2. Generate synthetic per-house load + voltage data
# ---------------------------------------------------------
# Household-scale power draw (kW-ish, in per-unit), NOT aggregated-bus scale.
# Real single-house peak demand is roughly 1-8 kW; here expressed in a
# normalized per-unit range appropriate for a small LV feeder base power.
#
# Two house-specific binary features are added since individual houses
# (unlike aggregated buses) plausibly differ in equipment:
#   - has_solar: rooftop PV, which can offset or reverse net load
#   - has_ev: EV charger, which adds a large, spiky load
#
# Real project: replace with actual smart-meter / AMI data (e.g. your
# Ausgrid dataset) instead of this synthetic generator.
def generate_synthetic_house_data(num_samples=500, N=N, seed=0):
    rng = np.random.default_rng(seed)

    # Fixed per-house attributes (same across all samples/snapshots)
    has_solar = (rng.random(N) < 0.3).astype(np.float32)
    has_ev = (rng.random(N) < 0.2).astype(np.float32)
    has_solar[0], has_ev[0] = 0.0, 0.0  # transformer node has neither

    X_list, Y_list = [], []
    for _ in range(num_samples):
        # Base household active/reactive load, small per-unit range
        P = rng.uniform(0.01, 0.06, size=N).astype(np.float32)   # baseline consumption
        Q = rng.uniform(0.005, 0.02, size=N).astype(np.float32)

        # Solar PV can offset or reverse net active power at that house
        solar_gen = has_solar * rng.uniform(0.0, 0.08, size=N).astype(np.float32)
        P = P - solar_gen

        # EV charging adds a large, occasionally-active spike
        ev_active = has_ev * (rng.random(N) < 0.4)
        P = P + ev_active * rng.uniform(0.03, 0.09, size=N).astype(np.float32)

        P[0], Q[0] = 0.0, 0.0  # transformer node itself has no injection

        # Simplified synthetic voltage-drop relationship along the tree,
        # accumulated by graph distance from the transformer (not real
        # power-flow physics — swap in an actual LV power-flow solver's
        # output as labels for a real deployment).
        depth = compute_depth_from_root(branches, N)
        voltage = 1.0 - 0.01 * depth * (0.5 + np.abs(P)) - 0.005 * depth * np.abs(Q)
        voltage += rng.normal(0, 0.001, size=N)

        features = np.stack([P, Q, has_solar, has_ev], axis=1)  # shape (N, 4)
        X_list.append(features)
        Y_list.append(voltage.astype(np.float32))

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


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


X_all, Y_all = generate_synthetic_house_data(num_samples=500, N=N)

# Train/val/test split (by sample, not by node — each sample = one grid snapshot)
n_train, n_val = 350, 75
X_train, Y_train = X_all[:n_train], Y_all[:n_train]
X_val, Y_val     = X_all[n_train:n_train+n_val], Y_all[n_train:n_train+n_val]
X_test, Y_test   = X_all[n_train+n_val:], Y_all[n_train+n_val:]

print(f"Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ---------------------------------------------------------
# 3. GAT model (regression: predict voltage per house)
# ---------------------------------------------------------
class HouseGAT(nn.Module):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HouseGAT(in_channels=X_all.shape[-1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
edge_index = edge_index.to(device)

# ---------------------------------------------------------
# 4. Train / Eval loops (loop over grid snapshots)
# ---------------------------------------------------------
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

if __name__ == "__main__":
    for epoch in range(1, 51):
        train_loss = run_epoch(X_train, Y_train, train=True)
        if epoch % 5 == 0:
            with torch.no_grad():
                val_loss = run_epoch(X_val, Y_val, train=False)
            print(f"Epoch {epoch:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    with torch.no_grad():
        test_loss = run_epoch(X_test, Y_test, train=False)
    print(f"\nFinal Test MSE: {test_loss:.6f}")

    # Sanity check: show predicted vs actual voltage for one test snapshot
    with torch.no_grad():
        x_sample = torch.tensor(X_test[0]).to(device)
        y_sample = Y_test[0]
        pred = model(x_sample, edge_index).cpu().numpy()
        print("\nSample house voltages (transformer + first 5 houses):")
        print(f"Predicted: {pred[:6]}")
        print(f"Actual:    {y_sample[:6]}")
