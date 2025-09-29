import testing
from train import train
import importlib
import warnings
warnings.filterwarnings("ignore")
importlib.reload(testing)
import random

import sys
if 'testing' in sys.modules:
    del sys.modules['testing']


def in_loop_retraining(experiment = "first_firm_constant", firm_count=4, episodes_before_retrain=7, max_retrain=100):
    retrain_count = 0
    K = firm_count

    firm_policies = [f"SAC_portion_{firm_count}" for _ in range(K)]  # Initial policies for each firm
    alphas = [random.uniform(sys.float_info.epsilon, 1 - sys.float_info.epsilon) for _ in range(K)]
    output_data = []
    while True:
        if experiment == "first_firm_constant":
            firm_policies[0] = f"SAC_portion_{firm_count}"

        # Test the model for episodes_before_retrain episodes
        config = {
            "simulator.name": "multi_macro",
            # "model.name": ["sac", "equal_distribution", "random"],
            "model.name": ["sac"],
            "simulator.city": "nyc_brooklyn",
            "model.cplexpath": None,
            "model.test_episodes": 7,
            "model.checkpoint_path": firm_policies,
            "model.alpha": alphas,
            "simulator.reuse_no_control": False,
            "simulator.firm_count": firm_count,
            "simulator.agents_know_partial_demand": True,
            "simulator.constant_vehicle_count": True,
            "simulator.demand_filter_type": "equal" ,
            "simulator.pricing_model": "equal",
            "simulator.initial_vehicle_distribution": "random",
            "model.loop_number": retrain_count
        }
        data = testing.multi_test(config)
        # test_results, data_files = testing.test_approach(cfg, env, parser, device, loop_number=retrain_count, name="sac")

        if retrain_count >= max_retrain:
            break
        # Retrain model for each agent
        firm_policies = [f"SAC_portion_4_firm_{k}_loop_{retrain_count+1}" for k in range(K)]
        for k in range(K):
            if experiment == "first_firm_constant":
                if k == 0:
                    continue  # Don't retrain the first firm
            # Train a model based on new historical demand
            config = {
                "simulator.name": "macro",
                "model.name": "sac",
                "simulator.city": "nyc_brooklyn",
                "simulator.demand": f"historical_demand_sac_firm_{k}_{retrain_count}",
                "model.cplexpath": None,
                "model.checkpoint_path": firm_policies[k],  # Save new model to new file
                "model.max_episodes": 50, # Was 50
                "simulator.reuse_no_control": False,
                "simulator.firm_count": 1,
                "simulator.agents_know_partial_demand": True,
                "simulator.constant_vehicle_count": True,
                "simulator.demand_filter_type": "flow" 
            }
            train(config)
        output_data.append(data)
        print(f"Profits after retrain {retrain_count}: ", [data[0]["sac"][k][0] for k in range(K)])
        retrain_count += 1

    # Dump data
    import pickle
    from datetime import datetime
    now = datetime.now()

    # Format for filename
    filename_time = now.strftime("%Y%m%d_%H%M%S")
    with open(f'saved_files/competition_loop_data_{firm_count}_firms_{max_retrain}_retrain_{filename_time}.pkl', 'wb') as f:
        pickle.dump(output_data, f)

    # Plot results
    # Subplot per firm with Overall profit vs week line plot using sac, if this firm does not retrain, equal_dist, random, and no control
    # import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    # data = pickle.load(open('saved_files/competition_loop_data_4_firms_100_retrain.pkl', 'rb'))
    # print(data)
    x = [[elem[0]["sac"][k][0] for elem in output_data] for k in range(4)]
    for k in range(4):
        plt.plot(np.arange(len(x[k])), x[k], label=f'Firm {k} SAC')
    plt.xlabel('Retrain Iteration')
    plt.ylabel('Overall Profit')
    plt.title('Overall Profit vs Retrain Iteration for Each Firm')

    plt.show()

def replot(filename):
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    data = pickle.load(open(filename, 'rb'))
    print(data)
    x = [[elem[0]["sac"][k][0] for elem in data] for k in range(4)]
    for k in range(4):
        plt.plot(np.arange(len(x[k])), x[k], label=f'Firm {k} SAC')
    plt.xlabel('Retrain Iteration')
    plt.ylabel('Overall Profit')
    plt.title('Overall Profit vs Retrain Iteration for Each Firm')

    plt.show()


if __name__ == "__main__":
    in_loop_retraining(experiment="first_firm_constant", firm_count=4, episodes_before_retrain=7, max_retrain=10)