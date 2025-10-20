from train import train
config = {
    "simulator.name": "macro",
    "model.name": "sac",
    "simulator.city": "san_francisco",
    "model.cplexpath": None, 
    "simulator.pricing_model": "quasi_cournot",
    "simulator.alpha": 0.3*14/4,
    "simulator.observation_model": "include_adjusted_revenue",
    "model.checkpoint_path": "SAC_SF_quasi_cournot_aggressive_price_obs_change_ent0_05",
    "model.max_episodes": 4000,
    "model.alpha": 0.05,
    "model.wandb": True,
}
train(config)