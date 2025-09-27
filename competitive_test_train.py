import testing
from train import train
import importlib
import warnings
warnings.filterwarnings("ignore")
importlib.reload(testing)

import sys
if 'testing' in sys.modules:
    del sys.modules['testing']


def in_loop_retraining(firm_count=4, episodes_before_retrain=7, max_retrain=10):
    retrain_count = 0
    K = firm_count
    firm_policies = [f"SAC_portion_{firm_count}" for _ in range(K)]  # Initial policies for each firm
    while True:
        # Test the model for episodes_before_retrain episodes
        config = {
            "simulator.name": "multi_macro",
            # "model.name": ["sac", "equal_distribution", "random"],
            "model.name": ["sac"],
            "simulator.city": "nyc_brooklyn",
            "model.cplexpath": None,
            "model.test_episodes": 1,
            "model.checkpoint_path": firm_policies,
            "simulator.reuse_no_control": False,
            "simulator.firm_count": firm_count,
            "simulator.agents_know_partial_demand": True,
            "simulator.constant_vehicle_count": True,
            "simulator.demand_filter_type": "equal" ,
            "simulator.pricing_model": "equal",
            "model.loop_number": retrain_count
        }
        data = testing.multi_test(config)
        # test_results, data_files = testing.test_approach(cfg, env, parser, device, loop_number=retrain_count, name="sac")

        if retrain_count >= max_retrain:
            break
        # Retrain model for each agent
        firm_policies = [f"SAC_portion_4_firm_{k}_loop_{retrain_count+1}" for k in range(K)]
        for k in range(K):
            # Train a model based on new historical demand
            config = {
                "simulator.name": "macro",
                "model.name": "sac",
                "simulator.city": "nyc_brooklyn",
                "simulator.demand": f"historical_demand_sac_firm_{k}_{retrain_count}",
                "model.cplexpath": None,
                "model.checkpoint_path": firm_policies[k],  # Save new model to new file
                "model.max_episodes": 50,
                "simulator.reuse_no_control": False,
                "simulator.firm_count": 1,
                "simulator.agents_know_partial_demand": True,
                "simulator.constant_vehicle_count": True,
                "simulator.demand_filter_type": "flow" 
            }
            train(config)
        retrain_count += 1


    # Plot results
    # Subplot per firm with Overall profit vs week line plot using sac, if this firm does not retrain, equal_dist, random, and no control


if __name__ == "__main__":
    in_loop_retraining(firm_count=4, episodes_before_retrain=7, max_retrain=10)