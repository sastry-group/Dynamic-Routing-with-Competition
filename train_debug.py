# from train import train
# config = {
#     "simulator.name": "multi_macro",
#     "model.name": "sac",
#     "simulator.city": "nyc_brooklyn",
#     "model.cplexpath": None, 
#     "model.checkpoint_path": "SAC_portion_2_test",
#     "model.max_episodes": 50,
#     "simulator.reuse_no_control": False,
#     "simulator.firm_count": 2,
#     "simulator.agents_know_partial_demand": True,
#     "simulator.constant_vehicle_count": True,
#     "simulator.demand_filter_type": "portion" 
# }
# train(config)


from train import train
import numpy as np

alphas = [0.5, 0.6, 0.3, 0.8]
alpha_choice = np.random.choice(alphas)
config = {
                "simulator.name": "multi_macro",
                "model.name": "sac",
                "simulator.city": "nyc_brooklyn",
                # "simulator.demand": f"historical_demand_sac_firm_{k}_{retrain_count}",
                "model.cplexpath": None,
                "model.checkpoint_path": f"SAC_initial_firm4_alpha{alpha_choice}",  # Save new model to new file
                "model.max_episodes": 50, # Was 50
                "model.wandb": False,
                "simulator.reuse_no_control": False,
                "simulator.firm_count": 1,
                "simulator.agents_know_partial_demand": True,
                "simulator.constant_vehicle_count": True,
                "simulator.pricing_model": "cournot",
                "simulator.demand_filter_type": "flow",
                "model.alpha": alpha_choice
            }
train(config)