from train import train
config = {
    "simulator.name": "macro",
    "model.name": "sac",
    "simulator.city": "nyc_brooklyn",
    "model.cplexpath": None, 
    "simulator.pricing_model": "quasi_cournot",
    "simulator.alpha": 0.7*14/4,
    "simulator.observation_model": "include_adjusted_revenue",
    "model.checkpoint_path": "SAC_NYC_quasi_cournot_aggressive_price_0_7_obs_change_ent0_3",
    "model.max_episodes": 10000,
    "model.alpha": 0.3,
    "model.wandb": True,
}
train(config)