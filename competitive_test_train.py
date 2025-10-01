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

     # Initial policies for each firm
    # alphas = [random.uniform(sys.float_info.epsilon, 1 - sys.float_info.epsilon) for _ in range(K)]
    alphas = [0.5, 0.6, 0.3, 0.8]
    firm_policies = [f"SAC_initial_firm4_alpha{alphas[i]}" for i in range(K)] 
    output_data = []
    
    while True:
        if experiment == "first_firm_constant":
            firm_policies[0] = f"SAC_initial_firm4_alpha{alphas[0]}"

        # Test the model for episodes_before_retrain episodes
        config = {
            "simulator.name": "multi_macro",
            # "model.name": ["sac", "equal_distribution", "random"],
            "model.name": ["sac"],
            "simulator.city": "nyc_brooklyn",
            "model.cplexpath": None,
            "model.test_episodes": episodes_before_retrain,
            "model.checkpoint_path": firm_policies,
            "model.alpha": alphas, # I think we should move this to simulator.alphas
            "simulator.reuse_no_control": False,
            "simulator.firm_count": firm_count,
            "simulator.agents_know_partial_demand": True,
            "simulator.constant_vehicle_count": True,
            "simulator.demand_filter_type": "price_based" ,
            "simulator.pricing_model": "cournot",
            "simulator.initial_vehicle_distribution": "random",
            "model.loop_number": retrain_count,
            "simulator.competition": True     # this is a new thing i added for competition, this is what toggled the multi in test_approach
        }
        data = testing.multi_test(config)
        # test_results, data_files = testing.test_approach(cfg, env, parser, device, loop_number=retrain_count, name="sac")

        output_data.append(data)
        print(f"Profits after retrain {retrain_count}: ", [data[0]["sac"][k][0] for k in range(K)])

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
                "model.max_episodes": 100, # Was 50
                "model.wandb": False,
                "simulator.reuse_no_control": False,
                "simulator.firm_count": 1,
                "simulator.agents_know_partial_demand": True,
                "simulator.constant_vehicle_count": True,
                "simulator.pricing_model": "cournot",
                "simulator.demand_filter_type": "flow",
                "simulator.alpha": alphas[k],   # for training use simulator
                "simulator.competition": True
            }
            train(config)
        # output_data.append(data)
        print(f"Profits after retrain {retrain_count}: ", [data[0]["sac"][k][0] for k in range(K)])
        retrain_count += 1

    # Dump data
    import pickle
    from datetime import datetime
    now = datetime.now()

    # Format for filename
    filename_time = now.strftime("%Y%m%d_%H%M%S")
    # alphas_file = f"saved_files/competition_loop_alphas_{firm_count}_firms_{max_retrain}_retrain_{filename_time}.pkl"
    alphas_file = f"saved_files/competition_loop_alphas.pkl"
    with open(f'saved_files/competition_loop_data_{firm_count}_firms_{max_retrain}_retrain_{filename_time}.pkl', 'wb') as f:
        pickle.dump(output_data, f)
    with open(alphas_file, "wb") as f:
        pickle.dump(alphas, f)

    # Plot results
    # Subplot per firm with Overall profit vs week line plot using sac, if this firm does not retrain, equal_dist, random, and no control
    # import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    # data = pickle.load(open('saved_files/competition_loop_data_4_firms_10_retrain_20250929_133001.pkl', 'rb'))
    # print(data)
    with open(alphas_file, "rb") as f:
        alphas = pickle.load(f)
    x = [[elem[0]["sac"][k][0] for elem in output_data] for k in range(4)]
    # x = [[elem[0]["sac"][k][0] for elem in data] for k in range(4)]
    window = 5
    for k in range(4):
        plt.plot(np.arange(len(x[k])), x[k], label=f'Firm {k} SAC, alpha={alphas[k]:.2f}')
        weights = np.ones(window) / window
        moving_avg = np.convolve(x[k], weights, mode='valid')
        plt.plot(np.arange(window-1, len(x[k])), moving_avg, linestyle=':', 
                label=f'Firm {k} Moving Avg ({window})')
        z = np.polyfit(np.arange(len(x[k])), x[k], 1)
        p = np.poly1d(z)
        plt.plot(np.arange(len(x[k])), p(np.arange(len(x[k]))), linestyle='--', label=f'Firm {k} Trend')
    plt.legend()
    plt.xlabel('Retrain Iteration')
    plt.ylabel('Overall Profit')
    plt.title('Overall Profit vs Retrain Iteration for Each Firm')

    plt.show()

def replot(filename):
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    data = pickle.load(open(filename, 'rb'))
    alphas_file = f"saved_files/competition_loop_alphas.pkl"
    alphas = pickle.load(open(alphas_file, "rb"))
    with open(alphas_file, "wb") as f:
        pickle.dump(alphas, f)
    print(data)
    x = [[elem[0]["sac"][k][0] for elem in data] for k in range(4)]
    for k in range(4):
        plt.plot(np.arange(len(x[k])), x[k], label=f'Firm {k} SAC, alpha={alphas[k]:.2f}')
        z = np.polyfit(np.arange(len(x[k])), x[k], 1)
        p = np.poly1d(z)
        plt.plot(np.arange(len(x[k])), p(np.arange(len(x[k]))), linestyle='--', label=f'Firm {k} Trend')
    plt.legend()
    plt.xlabel('Retrain Iteration')
    plt.ylabel('Overall Profit')
    plt.title('Overall Profit vs Retrain Iteration for Each Firm')

    plt.show()


if __name__ == "__main__":
    in_loop_retraining(experiment="first_firm_constant", firm_count=4, episodes_before_retrain=20, max_retrain=40)