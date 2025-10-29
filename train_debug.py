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

# alphas = [0.3, 0.5, 0.6, 0.8]
# alpha_choice = np.random.choice(alphas)
alpha_choice = 0.6
config = {
                "simulator.name": "multi_macro",
                "model.name": "sac",
                "simulator.city": "san_francisco",
                "simulator.demand": "san_francisco",
                # "simulator.demand": f"historical_demand_sac_firm_{k}_{retrain_count}",
                "model.cplexpath": None,
                "model.checkpoint_path": f"SAC_testNOM3_firm1_SF",  # Save new model to new file
                "model.max_episodes": 2000, # Was 50
                "model.wandb": True,
                "simulator.reuse_no_control": False,
                "simulator.firm_count": 1,
                "simulator.agents_know_partial_demand": True,
                "simulator.constant_vehicle_count": True,
                "simulator.competition": False,
                "simulator.pricing_model": "equal",
                "simulator.demand_filter_type": "flow",
                "simulator.alpha": alpha_choice,

            }
train(config)