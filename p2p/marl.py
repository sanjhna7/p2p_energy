"""GAT-MARL: train house agents with PPO and evaluate them.

    python -m p2p.marl                     train, then evaluate on `scenario`
    python -m p2p.marl --eval-only         evaluate an existing checkpoint

ALGORITHM
  Multi-agent PPO with parameter sharing: one ActorCritic (p2p/gat.py) acts
  for every house. Each agent's action is a battery power in [-1, 1] x the
  rated power, sampled from a Gaussian whose mean comes from that house's
  GAT embedding. Rewards are per agent (p2p/reward.py); advantages are
  computed per agent with GAE. The critic also sees a pooled community
  embedding during training only.

DATA SPLIT
  Training uses `training.start` / `training.days`, evaluation uses the
  `scenario` window; the two do not overlap. Same houses in both.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from p2p import baselines, metrics
from p2p.config import DEFAULT_CONFIG_PATH, Param, load_config
from p2p.env import CommunityEnv
from p2p.gat import ActorCritic, GraphContext
from p2p.run import default_out
from p2p.scenario import load_scenario


def training_scenario(cfg):
    t = cfg.training
    s = replace(cfg.scenario, start=Param(t.start.value, "SCENARIO"), days=Param(t.days.value, "SCENARIO"))
    train_cfg = replace(cfg, scenario=s)
    eval_start = pd.Timestamp(cfg.scenario.start.value)
    train_end = pd.Timestamp(t.start.value) + pd.Timedelta(days=t.days.value)
    eval_end = eval_start + pd.Timedelta(days=cfg.scenario.days.value)
    if pd.Timestamp(t.start.value) < eval_end and eval_start < train_end:
        raise ValueError("training window overlaps the evaluation window")
    return load_scenario(train_cfg)


class GATPolicy:
    """Wraps a trained model as a baseline-style policy: obs -> kW requests."""

    def __init__(self, model: ActorCritic, ctx: GraphContext, deterministic: bool = True):
        self.model, self.ctx, self.deterministic = model, ctx, deterministic

    @torch.no_grad()
    def act(self, obs):
        """(action in [-1, 1] before clipping, its log-prob, value, node features), per house."""
        x = torch.as_tensor(self.ctx.features(obs))
        mean, std, value = self.model(x, self.ctx.edge_index, 1, self.ctx.n_nodes, self.ctx.house_nodes)
        a = mean if self.deterministic else torch.normal(mean, std)
        logp = torch.distributions.Normal(mean, std).log_prob(a)
        return a[0].numpy(), logp[0].numpy(), value[0].numpy(), x.numpy()

    def __call__(self, obs):
        a, _, _, _ = self.act(obs)
        return np.clip(a, -1, 1) * self.ctx.p_scale


def collect(env, policy, cfg, rng):
    """One episode from a random start with random initial SoC."""
    t = cfg.training
    steps = t.episode_steps.value
    t0 = int(rng.integers(0, len(env.sc.timestamps) - steps + 1))
    obs = env.reset(t0=t0, horizon=steps, soc=rng.uniform(0.1, 0.9, env.n_houses))
    buf = {k: [] for k in ("x", "a", "logp", "v", "r")}
    while not env.done:
        a, logp, v, x = policy.act(obs)
        obs, r, _, _ = env.step(np.clip(a, -1, 1) * policy.ctx.p_scale)
        for k, val in zip(buf, (x, a, logp, v, r * t.reward_scale.value)):
            buf[k].append(val)
    return {k: np.asarray(v, dtype=np.float32) for k, v in buf.items()}


def gae(r, v, gamma, lam):
    """Per-agent advantages; the episode is cut off, so bootstrap with the last value."""
    adv = np.zeros_like(r)
    last = np.zeros(r.shape[1], dtype=np.float32)
    for t in reversed(range(len(r))):
        nxt = v[t + 1] if t + 1 < len(r) else v[t]
        delta = r[t] + gamma * nxt - v[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + v


def train(cfg, out: Path, log_every: int = 10):
    t = cfg.training
    torch.manual_seed(t.seed.value)
    rng = np.random.default_rng(t.seed.value)
    sc = training_scenario(cfg)
    env = CommunityEnv(sc, cfg)
    ctx = GraphContext(sc, cfg)
    model = ActorCritic()
    opt = torch.optim.Adam(model.parameters(), lr=t.learning_rate.value)
    policy = GATPolicy(model, ctx, deterministic=False)
    history = []

    for it in range(1, t.iterations.value + 1):
        eps = [collect(env, policy, cfg, rng) for _ in range(t.episodes_per_iteration.value)]
        adv, ret = zip(*(gae(e["r"], e["v"], t.gamma.value, t.gae_lambda.value) for e in eps))
        x = torch.as_tensor(np.concatenate([e["x"] for e in eps]))
        a = torch.as_tensor(np.concatenate([e["a"] for e in eps]))
        old_logp = torch.as_tensor(np.concatenate([e["logp"] for e in eps]))
        adv = torch.as_tensor(np.concatenate(adv))
        ret = torch.as_tensor(np.concatenate(ret))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n_steps = len(x)
        for _ in range(t.ppo_epochs.value):
            for idx in torch.randperm(n_steps).split(t.minibatch_steps.value):
                b = len(idx)
                mean, std, value = model(x[idx].reshape(b * ctx.n_nodes, -1), ctx.batch_edges(b),
                                         b, ctx.n_nodes, ctx.house_nodes)
                dist = torch.distributions.Normal(mean, std)
                ratio = (dist.log_prob(a[idx]) - old_logp[idx]).exp()
                clipped = ratio.clamp(1 - t.clip.value, 1 + t.clip.value)
                policy_loss = -torch.min(ratio * adv[idx], clipped * adv[idx]).mean()
                value_loss = (value - ret[idx]).pow(2).mean()
                loss = policy_loss + t.value_coef.value * value_loss - t.entropy_coef.value * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        mean_r = float(np.mean([e["r"].sum(0).mean() for e in eps])) / t.reward_scale.value
        history.append({"iteration": it, "episode_reward_per_house_aud": mean_r,
                        "std": float(model.log_std.exp())})
        if it % log_every == 0 or it == 1:
            print(f"iter {it:4d}  reward/house/episode {mean_r:8.3f} AUD  std {history[-1]['std']:.3f}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "gat_marl.pt")
    pd.DataFrame(history).to_csv(out / "training_curve.csv", index=False)
    return model


def evaluate(cfg, model, out: Path):
    sc = load_scenario(cfg)
    ctx = GraphContext(sc, cfg)
    policy = GATPolicy(model.eval(), ctx, deterministic=True)
    logs = {name: baselines.run(name, sc, cfg) for name in baselines.POLICIES}
    env = CommunityEnv(sc, cfg, p2p=True)
    obs = env.reset()
    while not env.done:
        obs, _, _, _ = env.step(policy(obs))
    logs["gat_marl"] = env.log_arrays()

    rows = {name: metrics.summarise(log, sc.dt_hours, baseline=logs["grid_only"]) for name, log in logs.items()}
    summary = pd.DataFrame(rows).T
    summary.index.name = "policy"
    summary.to_csv(out / "summary_with_marl.csv", float_format="%.4f")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(summary.T)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = args.out or default_out(cfg)
    if args.eval_only:
        model = ActorCritic()
        model.load_state_dict(torch.load(out / "gat_marl.pt"))
    else:
        model = train(cfg, out)
    evaluate(cfg, model, out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
