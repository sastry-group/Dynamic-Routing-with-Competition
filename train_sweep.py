
from train import train
import numpy as np
import wandb


base = {
    # put your non-sweep defaults here so they exist even without overrides
    "simulator.name": "multi_macro",
    "model.name": "sac",
    "simulator.city": "san_francisco",
    "simulator.demand": "san_francisco",
    "model.cplexpath": None,
    "model.max_episodes": 2000,
    "model.wandb": True,
    "simulator.reuse_no_control": False,
    "simulator.firm_count": 1,
    "simulator.agents_know_partial_demand": True,
    "simulator.constant_vehicle_count": True,
    "simulator.competition": False,
    "simulator.pricing_model": "equal",
    "simulator.demand_filter_type": "flow",
    # sweep params (will be overridden by sweep)
    "model.alpha": 0.3,
    "model.batch_size": 128,
    "model.hidden_size": 256,
    "model.rew_scale": 0.01,
}

# init once here; avoid re-init in train.py (see tweak below)
wandb.init(project="droute", config=base)
cfg = dict(wandb.config)


a  = cfg.get("model.alpha", "NA")
bs = cfg.get("model.batch_size", "NA")
hs = cfg.get("model.hidden_size", "NA")
rw = cfg.get("model.rew_scale", "NA")

ckpt_dir = f"SAC_{cfg['simulator.city']}_a{a}_bs{bs}_h{hs}_rw{rw}"
cfg["model.checkpoint_path"] = ckpt_dir  # hydra override: model.checkpoint_path

print(f"[sweep] checkpoint path: {ckpt_dir}")

train(cfg)