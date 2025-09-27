from sympy import ff
import hydra
from omegaconf import DictConfig
import torch
import json
import numpy as np
from copy import deepcopy
from hydra import initialize, compose
from src.algos.registry import get_model
import os 
from tqdm import trange
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from src.envs.sim.competition import CompetitionSim
from src.algos.reb_flow_solver import solveRebFlow
from src.envs.sim.multi_macro_env import Fleet
from src.misc.utils import dictsum


RUN_TIME = time.strftime("%Y%m%d-%H%M%S")

def setup_sumo(cfg):
    from src.envs.sim.sumo_env import Scenario, AMoD, GNNParser
    cfg.simulator.cplexpath = cfg.model.cplexpath
    if not cfg.simulator.directory:
        cfg.simulator.directory = f"{cfg.model.name}/{cfg.simulator.city}"
    cfg = cfg.simulator
    scenario_path = 'src/envs/data'
    cfg.sumocfg_file = f'{scenario_path}/{cfg.city}/{cfg.sumocfg_file}'
    cfg.net_file = f'{scenario_path}/{cfg.city}/{cfg.net_file}'
    demand_file = f'src/envs/data/scenario_lux{cfg.num_regions}.json'
    aggregated_demand = not cfg.random_od

    scenario = Scenario(
        num_cluster=cfg.num_regions, json_file=demand_file, aggregated_demand=aggregated_demand,
        sumo_net_file=cfg.net_file, acc_init=cfg.acc_init, sd=cfg.seed, demand_ratio=cfg.demand_ratio,
        time_start=cfg.time_start, time_horizon=cfg.time_horizon, duration=cfg.duration,
        tstep=cfg.matching_tstep, max_waiting_time=cfg.max_waiting_time
    )
    env = AMoD(scenario, cfg=cfg, beta=cfg.beta)
    parser = GNNParser(env, T=cfg.time_horizon, json_file=demand_file)
    return env, parser
    
def setup_macro(cfg):
    from src.envs.sim.macro_env import Scenario, AMoD, GNNParser
    with open("src/envs/data/macro/calibrated_parameters.json", "r") as file:
        calibrated_params = json.load(file)
    cfg.simulator.cplexpath = cfg.model.cplexpath
    if not cfg.simulator.directory:
        cfg.simulator.directory = f"{cfg.model.name}/{cfg.simulator.city}"
    cfg = cfg.simulator
    city = cfg.city
    scenario = Scenario(
        json_file=f"src/envs/data/macro/scenario_{city}.json",
        demand_ratio=calibrated_params[city]["demand_ratio"],
        json_hr=calibrated_params[city]["json_hr"],
        sd=cfg.seed,
        json_tstep=calibrated_params[city]["test_tstep"],
        tf=cfg.max_steps,
    )
    env = AMoD(scenario, cfg = cfg, beta = calibrated_params[city]["beta"])
    parser = GNNParser(env, T=cfg.time_horizon, json_file=f"src/envs/data/macro/scenario_{city}.json")
    return env, parser

def setup_multi_macro(cfg):
    from src.envs.sim.multi_macro_env import Scenario, AMoD, GNNParser
    with open("src/envs/data/macro/calibrated_parameters.json", "r") as file:
        calibrated_params = json.load(file)
    cfg.simulator.cplexpath = cfg.model.cplexpath
    if not cfg.simulator.directory:
        cfg.simulator.directory = f"{cfg.model.name}/{cfg.simulator.city}"
    cfg = cfg.simulator
    city = cfg.city
    demand_file = cfg.demand
    if cfg.constant_vehicle_count:
        supply_factor = cfg.firm_count
    else:
        supply_factor = 1
    cfg.demand_ratio = calibrated_params[city]["demand_ratio"]
    # cfg["json_tstep"] = calibrated_params[city]["test_tstep"]
    cfg.json_tsetp = calibrated_params[city]["test_tstep"]
    # supply_factor = 1 # For competition format, env now has full number of vehs
    if demand_file == city:
        json_file = f"src/envs/data/multi_macro/scenario_{city}.json"
    else:
        json_file = f"saved_files/{demand_file}.json"
    scenario = Scenario(
        # json_file=f"saved_files/scenario_{demand_file}.json",
        # json_file=f"src/envs/data/multi_macro/scenario_{city}.json",
        json_file=json_file,
        demand_ratio=cfg.demand_ratio,
        json_hr=calibrated_params[city]["json_hr"],
        sd=cfg.seed,
        json_tstep=cfg.json_tsetp,
        tf=cfg.max_steps,
        supply_factor=supply_factor,
        firm_count=cfg.firm_count,
        demand_filter_type = cfg.demand_filter_type,
        initial_vehicle_distribution=cfg.initial_vehicle_distribution,
        pricing_model=cfg.pricing_model

    )

    filename = RUN_TIME + "_supply_factor_" + str(supply_factor) + "_firm_count_" + str(cfg.firm_count) + "_dm_" + str(cfg.demand_filter_type)
    save_sampled_demand(scenario.tripAttr, filename)
    env = AMoD(scenario, cfg=cfg, beta=calibrated_params[city]["beta"])
    parser = GNNParser(env, T=cfg.time_horizon, json_file=f"src/envs/data/macro/scenario_{city}.json")
    return env, parser

def setup_model(cfg, env, parser, device):
    model_name = cfg.model.name
    
    if model_name == "sac" or model_name =="cql":
        from src.algos.sac import SAC
        model= SAC(env=env, input_size=cfg.model.input_size, cfg=cfg.model, parser=parser, device=device).to(device)
        model.load_checkpoint(path=f"ckpt/{cfg.model.checkpoint_path}_best.pth")
        return model
    elif model_name == "a2c":
        from src.algos.a2c import A2C
        model = A2C(env=env, input_size=cfg.model.input_size,cfg=cfg.model, parser=parser, device=device).to(device)
        model.load_checkpoint(path=f"ckpt/{cfg.model.checkpoint_path}_best.pth")
        return model
    elif model_name == "iql":
        from src.algos.iql import IQL
        model = IQL(env=env, input_size=cfg.model.input_size,cfg=cfg.model, parser=parser, device=device).to(device)
        model.load_checkpoint(path=f"ckpt/{cfg.model.checkpoint_path}.pth")
        return model
    elif model_name == "bc":
        from src.algos.bc import BC
        model = BC(env=env, input_size=cfg.model.input_size,cfg=cfg.model, parser=parser, device=device).to(device)
        model.load_checkpoint(path=f"ckpt/{cfg.model.checkpoint_path}.pth")
        return model
    else:
        model_class = get_model(model_name)
        
        model_kwargs = {
            "cplexpath": cfg.simulator.cplexpath,
            "directory": cfg.simulator.directory,
            "T": cfg.simulator.time_horizon,
            "policy_name": cfg.model.name
        }
        for key, value in cfg.model.items():
            if key not in model_kwargs:
                model_kwargs[key] = value
        return model_class(**model_kwargs)

def multi_test(input_config):
    '''
    for Colab tutorial
    '''
    # We always need a control!
    if "no_rebalancing" not in input_config["model.name"]:
        input_config["model.name"].append("no_rebalancing")
    multi_config = [{**copy.deepcopy(input_config), "model.name" : model} for model in input_config["model.name"]]

    data = [{}, {}, {}]
    for config in multi_config:

        if config["model.name"] == "no_rebalancing" and cfg.simulator.reuse_no_control:
            path = f'./src/envs/data/{cfg.simulator.name}/{cfg.simulator.city}_no_control_performance.json'
            #check if path exists
            if os.path.exists(path):
                with open(path, 'r') as f:
                    no_control_performance = json.load(f)
                no_reb_reward = no_control_performance['reward']
                no_reb_demand = no_control_performance['served_demand']
                no_reb_cost = no_control_performance['rebalancing_cost']
                data[0][config["model.name"]], data[1][config['model.name']] = (no_reb_reward, no_reb_demand, no_reb_cost), None
            continue

        with initialize(config_path="src/config"):
            cfg = compose(config_name="config", overrides= [f"{key}={value}" for key, value in config.items()])  # Load the configuration
            
        # Import simulator module based on the configuration
        simulator_name = cfg.simulator.name
        if simulator_name == "sumo":
            env, parser = setup_sumo(cfg)
        elif simulator_name == "macro":
            env, parser = setup_macro(cfg)
        elif simulator_name == "multi_macro":
            env, parser = setup_multi_macro(cfg)
        else:
            raise ValueError(f"Unknown simulator: {simulator_name}")
        
        use_cuda = not cfg.model.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        (profit, inflows), file_names = test_approach(cfg, env, parser, device, loop_number=config["model.loop_number"], name=f"_{config['model.name']}")
        data[0][config["model.name"]], data[1][config['model.name']], data[2][config['model.name']] = profit, inflows, file_names
        # profit, inflows= test_approach(cfg, env, parser, device)
        # data[0][config["model.name"]], data[1][config['model.name']] = profit, inflows

    # no_ctrl_cfg = ...
    # no_ctrl_env = ...
    # no_ctrl_parser = ...
    # no_ctrl_device = device = torch.device("cuda" if use_cuda else "cpu")
    # control_data = get_no_control_performance(cfg, no_ctrl_env, no_ctrl_parser, no_ctrl_device, use_saved_data=cfg.simulator.reuse_no_control)

    plot_multi_fleet_comparison(cfg, env, data)

    return data

def save_vehicle_distribution(acc, file_str=None):
    """
    Save vehicle distribution across all timesteps and regions.
    
    Parameters:
        acc (defaultdict): Nested dict [region][timestep] = count.
        sim_time (str): Timestamp string for filename.
    """
    filename = f"saved_files/vehicle_distribution_{file_str}.json"
    if not os.path.exists("saved_files"):
        os.makedirs("saved_files")
    
    full_snapshot = defaultdict(dict)
    for region, timestep_dict in acc.items():
        for timestep, value in timestep_dict.items():
            full_snapshot[str(timestep)][str(region)] = float(value)
    
    with open(filename, "w") as f:
        json.dump(full_snapshot, f, indent=2)

    print(f"Saved full vehicle distribution to {filename}")

def save_sampled_demand(tripAttr, filename=None):
    # save the sampled demand to a file

    
    if not os.path.exists('saved_files'):
        os.makedirs('saved_files')
    with open(f'saved_files/sample_demand_{filename}.json', 'w') as f:
        print(f'Saving sampled demand to saved_files/sample_demand_{filename}.json')
        json.dump(tripAttr, f, indent=4)
        

def test_approach(cfg, env, parser, device, loop_number=0, name=""):

    multi = cfg.simulator.firm_count 

    if not multi: # or multi <= 1:
        model = setup_model(cfg, env, parser, device)
        
        print(f'Testing model {cfg.model.name} on {cfg.simulator.name} environment')
    
        episode_reward, episode_served_demand, episode_rebalancing_cost, inflows = model.test(cfg.model.test_episodes, env)
        if cfg.simulator.constant_vehicle_count:
            supply_factor = cfg.simulator.firm_count
        else:
            supply_factor = 1
        file_str = RUN_TIME + "_" + str(cfg.model.name) + "_supply_factor_" + str(supply_factor) + "_firm_count_" + str(cfg.simulator.firm_count) + "_dm_" + str(cfg.simulator.demand_filter_type)
        save_vehicle_distribution(env.acc, file_str)


        print('Mean Episode Profit ($): ', np.mean(episode_reward))
        print('Mean Episode Served Demand- Proit($): ', np.mean(episode_served_demand))
        print('Mean Episode Rebalancing Cost($): ', np.mean(episode_rebalancing_cost))

        inflows = np.mean(inflows, axis=0)

        mean_reward = np.mean(episode_reward)
        mean_served_demand = np.mean(episode_served_demand)
        mean_rebalancing_cost = np.mean(episode_rebalancing_cost)

        mean_reward = round(mean_reward/1000,2)
        mean_served_demand = round(mean_served_demand/1000,2)
        mean_rebalancing_cost = round(mean_rebalancing_cost/1000,2)
        rl_means = [(mean_reward, mean_served_demand, mean_rebalancing_cost)]

        return rl_means, inflows
    
    # === Multi-fleet competition path ===
    K = cfg.simulator.firm_count
    # assert K == len(cfg.model.name), "simulator.firm_count must match len(model.name)" for now the same, just sac
    device = torch.device("cpu")
    test_episodes = cfg.model.test_episodes
    epochs = trange(test_episodes) 
    # 1) One Fleet env per firm from the SAME scenario
    with open("src/envs/data/macro/calibrated_parameters.json", "r") as file:
        calibrated_params = json.load(file)
    fleets = [Fleet(env, cfg, firm_id=f"firm_{k}") for k in range(K)]
    for f in fleets:
        if not hasattr(f, "region") and hasattr(f, "regions"):
            f.region = f.regions

    # 2) One model per firm, each attached to its own Fleet
    models = []
    for k in range(K):
        # Clone cfg but override model name/checkpoint for this firm
        # firm_cfg = deepcopy(cfg)
        m = setup_model(cfg, env, parser, device) # this could be env per fleet, unsure
        models.append(m)

    # 3) CompetitionSim setups the demand allocation
    sim = CompetitionSim(env.scenario, fleets)

    # 4) Drive synchronized episodes (since model.test assumes single-fleet)
    episode_rewards = [[] for _ in range(K)]
    episode_served  = [[] for _ in range(K)]
    episode_reb_cost= [[] for _ in range(K)]
    episode_inflows = [[] for _ in range(K)]
    # rl_means = [[] for _ in range(K)]
    # inflows = [[] for _ in range(K)]
    seeds = list(range(env.cfg.seed, env.cfg.seed + test_episodes+1))
    historical_demand_totals = [defaultdict(lambda: defaultdict(float)) for _ in range(K)]
    historical_price_totals = [defaultdict(lambda: defaultdict(float)) for _ in range(K)]

    for i_episode in epochs:
        eps_reward = [0] * K
        eps_served_demand = [0] * K
        eps_rebalancing_cost = [0] * K
        # eps_reward = [[] for _ in range(K)]
        # eps_served_demand = [[] for _ in range(K)]
        # eps_rebalancing_cost = [[] for _ in range(K)]
        # eps_rebalancing_veh = [[] for _ in range(K)]
        
        # inflow = [[] for _ in range(K)]
        # obs = [[] for _ in range(K)]
        # rew = [[] for _ in range(K)]
        # Set seed for reproducibility across different policies
        np.random.seed(seeds[i_episode])
        done = False
        env.reset(multi_agent=True)
        obs, rew = sim.reset()
        eps_reward = rew
        eps_served_demand = rew
        eps_rebalancing_cost = [0] * K


        while not done:
            obs_list = [parser.parse_obs((f.acc, f.time, f.dacc, env.demand)).to(device) for f in sim.fleets]

            env.time += 1
            reb_actions = []
            for k, f in enumerate(sim.fleets):
                if cfg.model.name == "sac":
                    a = models[k].select_action(obs_list[k], deterministic=True)
                    desiredAcc = {f.region[i]: int(a[i] * dictsum(f.acc, f.time + 1)) for i in range(len(f.region))}
                    # print(f"Firm {k} desiredAcc (sum {sum(desiredAcc.values())}): ", desiredAcc)
                    reb = solveRebFlow(f, [], desiredAcc, "None")
                    reb_actions.append(reb)
                elif cfg.model.name == "equal_distribution" or cfg.model.name == "random" or cfg.model.name == "no_rebalancing":
                    reb = models[k].select_action(sim.fleets[k])
                    reb_actions.append(reb)

            done, infos = sim.step(reb_actions)

            eps_reward = [eps_reward[k] + infos[k].get("profit", 0.0) - infos[k].get("rebalancing_cost", 0.0) for k in range(K)]
            eps_served_demand = [eps_served_demand[k] + infos[k].get("profit", 0.0) for k in range(K)]
            eps_rebalancing_cost = [eps_rebalancing_cost[k] + infos[k].get("rebalancing_cost", 0.0) for k in range(K)]

        for k in range(K):
            # info = infos[k]
            # net = info.get("profit", 0.0) - info.get("rebalancing_cost", 0.0)
            episode_rewards[k].append(eps_reward[k])
            episode_served[k].append(eps_served_demand[k])
            episode_reb_cost[k].append(eps_rebalancing_cost[k])

            inflow_vec = np.zeros(len(f.region))
            if reb_actions[k] is not None:
                for idx, (i, j) in enumerate(f.edges):
                    inflow_vec[j] += reb_actions[k][idx]
            episode_inflows[k].append(inflow_vec)

        for k in range(K):
            for t, demand_dict in sim.historical_demand.items():
                fleet_demand = demand_dict[k]
                for (i,j), d in fleet_demand.items():
                    historical_demand_totals[k][(i,j)][t] += d
            for t, price_dict in sim.historical_prices.items():
                fleet_prices = price_dict[k]
                for (i,j), p in fleet_prices.items():
                    historical_price_totals[k][(i,j)][t] += p

    # Average the historical demand over episodes
    historical_demand = deepcopy(historical_demand_totals)
    for k, f in enumerate(fleets):
        for (i,j), t_dict in historical_demand[k].items():
            for t, d in t_dict.items():
                historical_demand[k][(i,j)][t] = d / test_episodes / env.cfg.demand_ratio
    # print("Average historical demand per episode:", historical_demand)
    # Average the historical prices over episodes
    historical_prices = deepcopy(historical_price_totals)
    for k, f in enumerate(fleets):
        for (i,j), t_dict in historical_prices[k].items():
            for t, p in t_dict.items():
                historical_prices[k][(i,j)][t] = p / test_episodes
    # print("Average historical prices per episode:", historical_prices)

    json_file = f"src/envs/data/multi_macro/scenario_{cfg.simulator.city}.json"
    with open(json_file, 'r') as file:
        data = json.load(file)
    extra_save_data = {"nlat": data["nlat"], "nlon": data["nlon"], "totalAcc": data["totalAcc"], "rebTime": data["rebTime"], "topology_graph": data["topology_graph"]}

    # Write historical demand and prices data to files
    formatted_data = convert({"demand":historical_demand, "prices": historical_prices}, sim, json_start=env.scenario.json_start, json_tstep=cfg.simulator.json_tsetp, extra_data=extra_save_data)
    file_names = []
    for k in range(K):
        file_name = f'saved_files/historical_demand{name}_firm_{k}_{loop_number}.json'
        with open(file_name, 'w') as f:
            json.dump(formatted_data[k], f, indent=4)
        print(f'Saved historical demand for firm {k} to {file_name}')
        file_names.append(file_name)

    rl_means_per_fleet = []
    inflows_per_fleet = []
    for k in range(K):
        r  = np.sum(episode_rewards[k]) / max(1, test_episodes)
        sd = np.sum(episode_served[k])  / max(1, test_episodes)
        rc = np.sum(episode_reb_cost[k]) / max(1, test_episodes)
        rl_means_per_fleet.append((round(r / 1000, 2), round(sd / 1000, 2), round(rc / 1000, 2)))

        inflows_per_fleet.append(np.mean(np.stack(episode_inflows[k], axis=0), axis=0))

    return (rl_means_per_fleet, inflows_per_fleet), file_names


def convert(data, sim, json_start=0, json_tstep=1, extra_data=None):
    result = []
    for demand, prices, f in zip(data["demand"], data["prices"], sim.fleets):
        firm_result = []
        for (i,j) in f.edges:
        # for origin, destination in demand.keys():
            default_demand = sim.demand[i,j]
            default_prices = sim.price[i,j]
            if (i,j) not in demand:
                inner_demand = default_demand
                inner_prices = default_prices
            else:
                inner_demand = demand[(i, j)]
                inner_prices = prices[(i, j)]
            for time_stamp in default_demand.keys():
                if time_stamp not in inner_demand:
                    inner_demand[time_stamp] = default_demand[time_stamp]
                    inner_prices[time_stamp] = default_prices[time_stamp]
                firm_result.append({
                    "time_stamp": time_stamp*json_tstep + json_start,
                    "origin": i,
                    "destination": j,
                    "demand": inner_demand[time_stamp],
                    "travel_time": sim.travelTime[(i,j)][time_stamp],
                    "price": inner_prices[time_stamp]
                })
        firm_result = {"nlat": sim.scenario.N1, "nlon": sim.scenario.N2, "demand": firm_result}
        if extra_data is not None:
            firm_result.update(extra_data)
        result.append(firm_result)
        
        # result.append({"demand": firm_result, "nlon": data["nlon"], "nlat": data["nlat"]})
        # result["nlat"] = data["nlat"]
        #         self.N2 = data["nlon"]
    return result


def get_no_control_performance(cfg, env, parser, device, setup_model_fn=setup_model, use_saved_data=False):
    #check if no_control performance is saved
    path = f'./src/envs/data/{cfg.simulator.name}/{cfg.simulator.city}_no_control_performance.json'
    #check if path exists
    if os.path.exists(path) and use_saved_data:
        with open(path, 'r') as f:
            no_control_performance = json.load(f)
        no_reb_reward = no_control_performance['reward']
        no_reb_demand = no_control_performance['served_demand']
        no_reb_cost = no_control_performance['rebalancing_cost']
    else:
        print('No control performance not found. Calculating (this happens only the first time on a new environment)...')
        cfg_copy = cfg.copy()
        cfg_copy.model.name = 'no_rebalancing'
        model = setup_model_fn(cfg_copy, env, parser, device)
        no_reb_reward, no_reb_demand, no_reb_cost, _ = model.test(10, env)
        no_reb_reward = round(np.mean(no_reb_reward)/1000,2)
        no_reb_demand = round(np.mean(no_reb_demand)/1000,2)
        no_reb_cost = round(np.mean(no_reb_cost)/1000,2)
        no_control_performance = {'reward': no_reb_reward, 'served_demand': no_reb_demand, 'rebalancing_cost': no_reb_cost}
        print(f'No control performance calculated. Saving in {path}...')
        print(os.getcwd())
        with open(path, 'w') as f:
            json.dump(no_control_performance, f)
    
    return (no_reb_reward, no_reb_demand, no_reb_cost)
    
def plot_multi_fleet_comparison(cfg, env, comparison_data):
    # Function to add value labels on top of bars
    def add_value_labels(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    profit_data, inflows, _ = comparison_data
    labels = ['Overall Profit', 'Served Demand Profit', 'Rebalancing Cost']
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    num_bars = len(profit_data) + 1

    fig, axs = plt.subplots(nrows=1, ncols=cfg.simulator.firm_count, figsize=(15, 5))
    if cfg.simulator.firm_count == 1:
        axs = [axs]

    #fig, ax = plt.subplots(figsize=(8, 5))

    colors = sns.color_palette("hsv", len(profit_data) + 1)
    start_x = x - (num_bars-1)*width/2 
    for ind, (key, data) in enumerate(profit_data.items()):
        for (ax, d) in zip(axs, data):
            rects1 = ax.bar(start_x + ind*width, d, width, label=key, color=colors[ind])
            add_value_labels(rects1, ax) # Adding value labels to each bar

    for firm_num, ax in enumerate(axs):
        # rects2 = ax.bar(start_x + (ind+1)*width, control_data, width, label='No Control', color=colors[-1])
        # add_value_labels(rects2, ax)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Metrics')
        ax.set_ylabel('$, x10^3')
        if cfg.simulator.firm_count == 1:
            ax.set_title(f'Comparison on {cfg.simulator.city} Environment with 1 Firm')
        else:
            ax.set_title(f'Firm {firm_num+1}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()

    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # if cfg.simulator.city != 'nyc_brooklyn': 
    plt.show()
    a = 1 # just to have a breakpoint
    # else: 
    #     #plots for tutorial
    #     open_reqest = {0: 0,
    #         1: 414.0,
    #         2: 0,
    #         3: 0,
    #         4: 0,
    #         5: 49756.49999999998,
    #         6: 9948.600000000006,
    #         7: 98.99999999999999,
    #         8: 198.00000000000003,
    #         9: 881.9999999999998,
    #         10: 1232.9999999999993,
    #         11: 6492.600000000001,
    #         12: 23293.80000000004,
    #         13: 170.99999999999997}
        
    #     #open_reqest = {k: v / max(open_reqest.values()) for k,v in open_reqest.items()}

    #     #inflows = inflows / max(inflows)

    #     labels = range(14)
    #     x = np.arange(len(labels))  # the label locations
    #     width = 0.25  # the width of the bars

    #     r1 = np.arange(14)
    #     r2 = [x + width for x in r1]

    #     #fig, ax = plt.subplots(figsize=(8, 5))
    #     for key, data in inflows.items():
    #         ax2.bar(r2, data, width, label=f'Rebalancing Flows for {key}', color="#0072BD")
    #     ax3 = ax2.twinx()  # Create a second y-axis
    #     ax3.bar(r1, open_reqest.values(), width, label='Profit', color="#A2142F")

    #     # Add labels and title to the second plot
    #     ax2.set_xlabel('Regions')
    #     ax2.set_ylabel('Flows', color="#0072BD")
    #     ax3.set_ylabel('Profit', color="#A2142F")
    #     ax2.set_title('Comparison of Incoming Rebalancing Flows vs Profit')
    #     ax2.set_xticks(r1)
    #     ax2.set_xticklabels(labels)
    #     ax2.tick_params(axis='y', labelcolor="#0072BD")
    #     ax3.tick_params(axis='y', labelcolor="#A2142F")
    #     #ax2.legend()
    #     #ax3.legend()

    #     plt.tight_layout()  
    #     plt.show()


def plot_comparison(cfg, env, control_data, comparison_data):
    # Function to add value labels on top of bars
    def add_value_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    profit_data, inflows = comparison_data
    labels = ['Overall Profit', 'Served Demand Profit', 'Rebalancing Cost']
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    num_bars = len(profit_data) + 1

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    #fig, ax = plt.subplots(figsize=(8, 5))

    colors = sns.color_palette("hsv", len(profit_data) + 1)
    start_x = x - (num_bars-1)*width/2 
    for ind, (key, data) in enumerate(profit_data.items()):
        rects1 = ax1.bar(start_x + ind*width, data, width, label=key, color=colors[ind])
        add_value_labels(rects1) # Adding value labels to each bar
    rects2 = ax1.bar(start_x + (ind+1)*width, control_data, width, label='No Control', color=colors[-1])
    add_value_labels(rects2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('$, x10^3')
    if cfg.simulator.firm_count == 1:
        ax1.set_title(f'Comparison on {cfg.simulator.city} Environment with 1 Firm')
    else:
        ax1.set_title(f'Comparison on {cfg.simulator.city} Environment with {cfg.simulator.firm_count} Firms')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    #plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    if cfg.simulator.city != 'nyc_brooklyn': 
        plt.show()
    else: 
        #plots for tutorial
        open_reqest = {0: 0,
            1: 414.0,
            2: 0,
            3: 0,
            4: 0,
            5: 49756.49999999998,
            6: 9948.600000000006,
            7: 98.99999999999999,
            8: 198.00000000000003,
            9: 881.9999999999998,
            10: 1232.9999999999993,
            11: 6492.600000000001,
            12: 23293.80000000004,
            13: 170.99999999999997}
        
        #open_reqest = {k: v / max(open_reqest.values()) for k,v in open_reqest.items()}

        #inflows = inflows / max(inflows)

        labels = range(14)
        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars

        r1 = np.arange(14)
        r2 = [x + width for x in r1]

        #fig, ax = plt.subplots(figsize=(8, 5))
        for key, data in inflows.items():
            ax2.bar(r2, data, width, label=f'Rebalancing Flows for {key}', color="#0072BD")
        ax3 = ax2.twinx()  # Create a second y-axis
        ax3.bar(r1, open_reqest.values(), width, label='Profit', color="#A2142F")

        # Add labels and title to the second plot
        ax2.set_xlabel('Regions')
        ax2.set_ylabel('Flows', color="#0072BD")
        ax3.set_ylabel('Profit', color="#A2142F")
        ax2.set_title('Comparison of Incoming Rebalancing Flows vs Profit')
        ax2.set_xticks(r1)
        ax2.set_xticklabels(labels)
        ax2.tick_params(axis='y', labelcolor="#0072BD")
        ax3.tick_params(axis='y', labelcolor="#A2142F")
        #ax2.legend()
        #ax3.legend()

        plt.tight_layout()  
        plt.show()


def test(config):
    '''
    for Colab tutorial
    '''
    with initialize(config_path="src/config"):
        cfg = compose(config_name="config", overrides= [f"{key}={value}" for key, value in config.items()])  # Load the configuration
        
    # Import simulator module based on the configuration
    simulator_name = cfg.simulator.name
    if simulator_name == "sumo":
        env, parser = setup_sumo(cfg)
    elif simulator_name == "macro":
        env, parser = setup_macro(cfg)
    elif simulator_name == "multi_macro":
        env, parser = setup_multi_macro(cfg)
    else:
        raise ValueError(f"Unknown simulator: {simulator_name}")
    
    use_cuda = not cfg.model.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = setup_model(cfg, env, parser, device)
    
    print(f'Testing model {cfg.model.name} on {cfg.simulator.name} environment')
    episode_reward, episode_served_demand, episode_rebalancing_cost, inflows = model.test(cfg.model.test_episodes, env)
    # save_vehicle_distribution(env.acc, timestep=0, sim_time=time.strftime("%Y%m%d-%H%M%S"))

    print('Mean Episode Profit ($): ', np.mean(episode_reward))
    print('Mean Episode Served Demand- Proit($): ', np.mean(episode_served_demand))
    print('Mean Episode Rebalancing Cost($): ', np.mean(episode_rebalancing_cost))

    inflows = np.mean(inflows, axis=0)
    
    #check if no_control performance is saved
    path = f'./src/envs/data/{cfg.simulator.name}/{cfg.simulator.city}_no_control_performance.json'
    #check if path exists
    if os.path.exists(path):
        with open(path, 'r') as f:
            no_control_performance = json.load(f)
        no_reb_reward = no_control_performance['reward']
        no_reb_demand = no_control_performance['served_demand']
        no_reb_cost = no_control_performance['rebalancing_cost']
    else:
        print('No control performance not found. Calculating (this happens only the first time on a new environment)...')
        cfg_copy = cfg.copy()
        cfg_copy.model.name = 'no_rebalancing'
        model = setup_model(cfg_copy, env, parser, device)
        no_reb_reward, no_reb_demand, no_reb_cost, _ = model.test(10, env)
        no_reb_reward = round(np.mean(no_reb_reward)/1000,2)
        no_reb_demand = round(np.mean(no_reb_demand)/1000,2)
        no_reb_cost = round(np.mean(no_reb_cost)/1000,2)
        no_control_performance = {'reward': no_reb_reward, 'served_demand': no_reb_demand, 'rebalancing_cost': no_reb_cost}
        print(f'No control performance calculated. Saving in {path}...')
        print(os.getcwd())
        with open(path, 'w') as f:
            json.dump(no_control_performance, f)

    mean_reward = np.mean(episode_reward)
    mean_served_demand = np.mean(episode_served_demand)
    mean_rebalancing_cost = np.mean(episode_rebalancing_cost)

    mean_reward = round(mean_reward/1000,2)
    mean_served_demand = round(mean_served_demand/1000,2)
    mean_rebalancing_cost = round(mean_rebalancing_cost/1000,2)
    labels = ['Overall Profit', 'Served Demand Profit', 'Rebalancing Cost']
    rl_means = [mean_reward, mean_served_demand, mean_rebalancing_cost]
    
    no_control = [no_reb_reward, no_reb_demand, no_reb_cost]
    
    import matplotlib.pyplot as plt
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    #fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax1.bar(x - width/2, rl_means, width, label=cfg.model.name, color="#0072BD")
    rects2 = ax1.bar(x + width/2, no_control, width, label='No Control', color="#A2142F")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('$, x10^3')
    ax1.set_title(f'Comparison of {cfg.model.name} vs No Control')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    
    # Function to add value labels on top of bars
    def add_value_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Adding value labels to each bar
    add_value_labels(rects1)
    add_value_labels(rects2)

    #plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    if cfg.simulator.city != 'nyc_brooklyn': 
        plt.show()
    else: 
        #plots for tutorial
        open_reqest = {0: 0,
            1: 414.0,
            2: 0,
            3: 0,
            4: 0,
            5: 49756.49999999998,
            6: 9948.600000000006,
            7: 98.99999999999999,
            8: 198.00000000000003,
            9: 881.9999999999998,
            10: 1232.9999999999993,
            11: 6492.600000000001,
            12: 23293.80000000004,
            13: 170.99999999999997}
        
        #open_reqest = {k: v / max(open_reqest.values()) for k,v in open_reqest.items()}

        #inflows = inflows / max(inflows)

        labels = range(14)
        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars

        r1 = np.arange(14)
        r2 = [x + width for x in r1]

        #fig, ax = plt.subplots(figsize=(8, 5))
        ax2.bar(r2, inflows, width, label='Rebalancing Flows', color="#0072BD")
        ax3 = ax2.twinx()  # Create a second y-axis
        ax3.bar(r1, open_reqest.values(), width, label='Profit', color="#A2142F")

        # Add labels and title to the second plot
        ax2.set_xlabel('Regions')
        ax2.set_ylabel('Flows', color="#0072BD")
        ax3.set_ylabel('Profit', color="#A2142F")
        ax2.set_title('Comparison of Incoming Rebalancing Flows vs Profit')
        ax2.set_xticks(r1)
        ax2.set_xticklabels(labels)
        ax2.tick_params(axis='y', labelcolor="#0072BD")
        ax3.tick_params(axis='y', labelcolor="#A2142F")
        #ax2.legend()
        #ax3.legend()

        plt.tight_layout()  
        plt.show()

@hydra.main(version_base=None, config_path="src/config/", config_name="config")
def main(cfg: DictConfig):
   
    # Import simulator module based on the configuration
    simulator_name = cfg.simulator.name
    if simulator_name == "sumo":
        env, parser = setup_sumo(cfg)

    elif simulator_name == "macro":
        env, parser = setup_macro(cfg)

    elif simulator_name == "multi_macro":
        env, parser = setup_multi_macro(cfg)
    else:
        raise ValueError(f"Unknown simulator: {simulator_name}")

    use_cuda = not cfg.model.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = setup_model(cfg, env, parser, device)
    
    print('Testing...')
    episode_reward, episode_served_demand, episode_rebalancing_cost, episode_inflows = model.test(cfg.model.test_episodes, env)

    print('Mean Episode Profit ($): ', np.mean(episode_reward), 'Std Episode Reward: ', np.std(episode_reward))
    print('Mean Episode Served Demand($): ', np.mean(episode_served_demand), 'Std Episode Served Demand: ', np.std(episode_served_demand))
    print('Mean Episode Rebalancing Cost($): ', np.mean(episode_rebalancing_cost), 'Std Episode Rebalancing Cost: ', np.std(episode_rebalancing_cost))

    ##TODO: ADD VISUALIZATION

if __name__ == "__main__":
    main()
    