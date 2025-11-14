
from train import train
import numpy as np
import wandb


base = {
    # put your non-sweep defaults here so they exist even without overrides
    "simulator.name": "macro",
    "model.name": "sac",
    "simulator.city": "nyc_brooklyn",
    "model.cplexpath": None,
    "model.max_episodes": 10000,
    "model.wandb": True,
    "simulator.observation_model": "include_adjusted_revenue",
    "simulator.pricing_model": "quasi_cournot",
    "simulator.alpha": 0.7*14/4,
    # sweep params (will be overridden by sweep)
    "model.alpha": 0.3,
    "model.batch_size": 128,
    "model.hidden_size": 256,
    "model.only_q_steps": 0
}

# init once here; avoid re-init in train.py (see tweak below)
wandb.init(project="droute", config=base)
cfg = dict(wandb.config)


a  = cfg.get("model.alpha", "NA")
bs = cfg.get("model.batch_size", "NA")
hs = cfg.get("model.hidden_size", "NA")
oqs = cfg.get("model.only_q_steps", "NA")

ckpt_dir = f"SAC_{cfg['simulator.city']}_quasi_cournot_aggressive_price_0_7_obs_change_a{a}_bs{bs}_h{hs}_oqs{oqs}"
cfg["model.checkpoint_path"] = ckpt_dir  # hydra override: model.checkpoint_path

print(f"[sweep] checkpoint path: {ckpt_dir}")

train(cfg)