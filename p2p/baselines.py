"""Rule-based baselines the learned policy must beat.

Each policy maps an observation to battery power requests (kW, > 0
discharge). The battery model and safety layer limit what is executed.

  grid_only     batteries idle, no P2P: today's utility-only situation
  battery_only  greedy self-consumption, no P2P: what a house can do alone
  rule_p2p      greedy self-consumption + P2P at the MMR
"""

import numpy as np

from p2p.config import Config
from p2p.env import CommunityEnv
from p2p.scenario import Scenario


def idle(obs):
    return np.zeros_like(obs["soc"])


def self_consumption(obs):
    """Charge with any surplus, discharge to cover any deficit."""
    return obs["load_kw"] - obs["pv_kw"]


POLICIES = {
    "grid_only": (idle, False),
    "battery_only": (self_consumption, False),
    "rule_p2p": (self_consumption, True),
}


def run(policy_name: str, sc: Scenario, cfg: Config) -> dict:
    policy, p2p = POLICIES[policy_name]
    env = CommunityEnv(sc, cfg, p2p=p2p)
    obs = env.reset()
    while not env.done:
        obs, _, _, _ = env.step(policy(obs))
    return env.log_arrays()
