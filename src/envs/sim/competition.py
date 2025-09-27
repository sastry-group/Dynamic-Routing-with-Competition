# competitor.py
from collections import defaultdict
from copy import deepcopy
from src.misc.utils import dictsum
import math
import numpy as np


# class Allocator:
#     """

#     """
#     def __init__(self, fleets, rule="equal", eps=1e-9):
#         self.fleets = fleets
#         self.rule = rule
#         self.eps = eps
#         self.t = 0
#         self.firm_count = len(fleets)

#     def compute_demand(self, demand_global_t):
#         K = len(self.fleets)
#         demand_per_firm = [defaultdict(float) for _ in range(K)]
#         if self.rule == "equal":
#             for e, D in demand_global_t.items():
#                 share = D / K
#                 for k in range(K):
#                     demand_per_firm[k][e] = share
#             return demand_per_firm


#         # as a function of price 
#         origins = {i for (i, _) in demand_global_t.keys()}
#         weights = {i: [] for i in origins}
#         for i in origins:
#             accs = []
#             for f in self.fleets:
#                 acc_i = f.acc[i].get(self.t+1, f.acc[i].get(self.t, 0.0))
#                 accs.append(max(0.0, acc_i))
#             S = sum(accs) + self.eps
#             weights[i] = [a / S for a in accs]

#         for (i, j), D in demand_global_t.items():
#             for k in range(K):
#                 demand_per_firm[k][(i, j)] = D * weights[i][k]
#         return demand_per_firm

#     def step(self):
#         self.t += 1

#     def compute_price(self, i, j, t, base_price, pricing_model):
#         # print(pricing_model)
#         # model: "cournot", "bertrand", "exogenous"
#         if pricing_model == "cournot":
#             try:
#                 q_total = sum(self.fleets[f].acc[i][t] for f in range(self.firm_count))
#             except KeyError:
#                 q_total = sum(self.fleets[f].acc[i] for f in range(self.firm_count))
#             if q_total <= 0:
#                 return base_price 

#             # supply, number of initial vehicles (constant right now)
#             # chekcing the q_total with total_Supplu
#             a = base_price 
#             alpha = 0.1
#             b = alpha * a * (1 / q_total)
#             cournot_price = a - b * q_total
#             # print(supply, q_total, p) # or current planned quantity
#             # print(f"Cournot price for edge ({i},{j}) at time {t}: {cournot_price}, and p,q: {p}, {q_total}")
#             return cournot_price
#         elif pricing_model == "bertrand":
#             return base_price
#         else:
#             return base_price




class CompetitionSim:
    """
    Orchestrates multi-fleet competition with shared demand+prices,
    per-fleet independent LP matching and rebalancing.
    You pass in:
      - scenario: provides demand_input, p, demandTime, rebTime, tf
      - fleets:   list of your per-fleet envs (refactor of AMoD with private state)
      - allocator: splits global demand into per-fleet demand caps each step
    This class drives the episode and queries each model for actions.
    """
    def __init__(self, scenario, fleets):
        # self.rule="equal"
        self.rule = "cournot"
        self.pricing_model = "cournot"
        self.eps=1e-9
        self.scenario = scenario
        self.travelTime = self.scenario.demandTime
        self.fleets = fleets
        # self.allocator = allocator
        self.t = 0
        self.time = 0
        self.tf = scenario.tf
        self.demand = defaultdict(dict) # demand
        self.price = defaultdict(dict) # price
        for i,j,t,d,p in scenario.tripAttr: # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i,j][t] = d
            # print(d,i,j,t,p)
            # self.price[i,j][t] = self.compute_price(i, j, t, p, pricing_model=self.pricing_model)
            self.price[i,j][t] = p
            # self.depDemand[i][t] += d
            # self.arrDemand[i][t+self.demandTime[i,j][t]] += d
        # print(f"Initial demand: {self.demand}")
        self.G = scenario.G
        self.edges = []
        self.beta = 0.3 # sensitivity parameter for price computation
        for i in self.G:
            self.edges.append((i,i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.historical_demand = {} # store historical demand for use in training loop
        self.historical_prices = {} # store historical prices for use in training loop

    def _market_publish(self):
        # shared total demand & price at time t
        # D = {(i, j): self.scenario.demand_input[i, j].get(self.t, 0.0)
        #      for (i, j) in self.scenario.edges}
        t = self.time
        D = {(i, j): self.demand[i, j].get(t, 0.0)
             for (i, j) in self.demand if self.demand[i,j][t]>1e-3}
        P = {(i, j): self.price[i, j].get(t, 0.0)
             for (i, j) in self.price if self.demand[i,j][t]>1e-3}
        T_pax = {(i, j): self.travelTime[i, j].get(t, 0)
                 for (i, j) in self.travelTime if self.demand[i,j][t]>1e-3}
        return D, P, T_pax

    def reset(self):
        self.time = 0
        # self.allocator.t = 0
        # Re-init each fleet’s private state (use your existing reset logic)
        for f in self.fleets:
            # do a light reset that preserves scenario but resets fleet state
            f.time = 0
            for n in f.regions:
                f.acc[n] = {0: f.G.nodes[n]['accInit']}
                f.dacc[n].clear()
            for e in f.G.edges:
                f.rebFlow[e].clear()
            for e in f.edges:
                f.paxFlow[e].clear()
        
        for i in self.G:
            self.edges.append((i,i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.demand = defaultdict(dict) # demand
        self.price = defaultdict(dict) # price

        # we might need to modify this
        # tripAttr = self.scenario.get_random_demand(reset=True)
        # self.regionDemand= defaultdict(dict)
        for i,j,t,d,p in self.scenario.tripAttr: # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i,j][t] = d # self.allocator.compute_demand(tripAttr)
            self.price[i,j][t] = p # self.allocator.compute_price(i,j,t,p,self.cfg.pricing_model)
            # self.demand[i,j][t] = d
            # self.price[i,j][t] = p
            # if t not in self.regionDemand[i]:
            #     self.regionDemand[i][t] = 0
            # else:
            #     self.regionDemand[i][t] +=d
        
        # 1) market info
        D, P, T_pax = self._market_publish()

        # 2) deterministic demand split (no bidding)
        # demand = self.allocator.compute_demand(D)
        prices = self.compute_price_per_t(D)
        demand = self.compute_demand_per_t(D, prices)

        self.historical_demand = {}
        if self.time in self.historical_demand.keys():
            raise ValueError("Demand for time t already exists in historical_demand")
        self.historical_demand[self.time] = demand
        self.historical_prices = {}
        if self.time in self.historical_prices.keys():
            raise ValueError("Prices for time t already exists in historical_prices")
        self.historical_prices[self.time] = prices
        # self.compute_price_per_t()
        
        self.time = 0
        obs = []
        paxreward = []
        done = []
        info = []
        for f, d, p in zip(self.fleets, demand, prices):
            for i,j in self.G.edges:
                f.rebFlow[i,j] = defaultdict(float)
                f.paxFlow[i,j] = defaultdict(float)            
            for n in self.G:
                f.acc[n][0] = self.G.nodes[n]['accInit']
                f.dacc[n] = defaultdict(float) 
            t = self.time
            # for i,j in self.demand:
            #     f.servedDemand[i,j] = defaultdict(float)
            # TODO: define states here
            f.demand = defaultdict(float)
            f.obs = (f.acc, f.time, f.dacc, f.demand)
            obs.append(f.obs)
            # paxreward.append(0)
            # done.append(f.done)
            info.append(f.info)
            f.obs, rew, f.done, f.info = f.pax_step(d, p, self.travelTime)
            print(f"Pax: {rew} -- Total: {rew}")
            paxreward.append(rew)
            f.reward = 0
        done = (self.tf == self.time+1)

        return obs, paxreward


    def step(self, reb_actions_by_fleet):
        """
        reb_actions_by_fleet: list[dict[(i,j)->float]] from each model this step
        """

        # 3) apply private rebalancing from the agents
        fleets_info = []
        for f, reb in zip(self.fleets, reb_actions_by_fleet):
            fleet_info = {}
            obs, rebreward, done, info = f.reb_step(reb)
            fleet_info['rebalancing_cost'] = -rebreward
            fleets_info.append(fleet_info)

        self.time += 1

        # 1) market info
        D, P, T_pax = self._market_publish()

        # 2) deterministic demand split (no bidding)
        # demand = self.allocator.compute_demand(D)
        prices = self.compute_price_per_t(D)
        demand = self.compute_demand_per_t(D, prices)

        if self.time in self.historical_demand.keys():
            raise ValueError("Demand for time t already exists in historical_demand")
        self.historical_demand[self.time] = demand
        if self.time in self.historical_prices.keys():
            raise ValueError("Prices for time t already exists in historical_prices")
        self.historical_prices[self.time] = prices

        # 4) each fleet solves its own pax LP with its cap + shared price
        #    Implemented as Fleet.match_with_caps(caps, P) returning {(i,j):flow}
        # matched_list = []
        pax_return_vars = []
        for f, fleet_demand, fleet_price, fleet_info in zip(self.fleets, demand, prices, fleets_info):
            # flows = f.matching(fleet_demand, fleet_price)  # your LP (adapted from matching_pulp)
            # f.apply_pax(flows, T_pax, P)        # updates revenue/costs/private state
            # matched_list.append(flows)
            obs, reward, done, info = f.pax_step(deepcopy(fleet_demand), deepcopy(fleet_price), self.travelTime)
            # pax_return_vars.append((obs, reward, done, info))
            fleet_info['profit'] = reward


        # 5) arrivals + time advance
        # for f in self.fleets:
        #     f.advance()
        # self.t += 1
        # self.allocator.step()
        print(f"Rebalancing: {rebreward} -- Pax: {reward} -- Total: {reward+rebreward}")

        done = (self.tf == self.time + 1)
        # collect per-fleet per-step info if needed
        # infos = [f.info.copy() for f in self.fleets]
        return done, fleets_info

    def compute_demand_per_t(self, demand_global_t, prices):
        K = len(self.fleets)
        time = self.time
        rng = np.random.default_rng() 
        global_demand_input = self.scenario.demand_input
        demand_per_firm = [defaultdict(float) for _ in range(K)]
        if self.rule == "equal":
            for e, D in demand_global_t.items():
                share = D / K
                for k in range(K):
                    demand_per_firm[k][e] = share
            return demand_per_firm

        # as a function of price 
        # for (i, j), D in demand_global_t.items():
        #     denom = sum([math.exp(-self.beta * prices[k][i,j]) for k in range(K)]) + self.eps
        #     for k in range(K):
        #         numer = math.exp(-self.beta * prices[k][i,j])
        #         demand_per_firm[k][(i, j)] = D * (numer / denom)
        for (i, j), data in global_demand_input.items():
            D = data.get(time, 0.0)
            weights = [math.exp(-self.beta * prices[k][i, j]) for k in range(K)]
            denom = sum(weights)
            probs = [w / denom for w in weights]
            if D <= 0:
                continue
        
            counts = rng.multinomial(int(D), probs)
            for k in range(K):
                demand_per_firm[k][(i, j)] = counts[k]

            # denom = sum([math.exp(-self.beta * prices[k][i,j]) for k in range(K)]) + self.eps
            # for k in range(K):
            #     numer = math.exp(-self.beta * prices[k][i,j])
            #     demand_per_firm[k][(i, j)] = D * (numer / denom)



        return demand_per_firm
    
    def compute_price_per_t(self, demand_global_t):
        fleet_prices = []
        for f in self.fleets:
            price_t = defaultdict(float)
            for (i,j) in demand_global_t.keys():
                if demand_global_t[i,j]<1e-3:
                    continue
                base_price = self.price[i,j].get(self.time, 0.0)
                price_t[i,j] = f.compute_price(i, j, self.time, base_price, pricing_model=f.pricing_model)
            fleet_prices.append(price_t)
            f.price = price_t
        return fleet_prices

    # def compute_price(self, i, j, t, alpha, base_price):
    #     # print(pricing_model)
    #     # model: "cournot", "bertrand", "exogenous"
    #     if self.pricing_model == "cournot":
    #         # Prices are being treated as equal per i across all destinations j
    #         try:
    #             q_total = sum(self.fleets[f].acc[i][t] for f in range(self.firm_count))
    #         except KeyError:
    #             q_total = sum(self.fleets[f].acc[i] for f in range(self.firm_count))
    #         if q_total <= 0:
    #             return base_price 

    #         a = 2 * base_price  # Should this really be 2x?
    #         b = alpha * a * (1 / q_total)
    #         cournot_price = a - b * q_total
    #         # print(supply, q_total, p) # or current planned quantity
    #         # print(f"Cournot price for edge ({i},{j}) at time {t}: {cournot_price}, and p,q: {p}, {q_total}")
    #         return cournot_price
    #     else:
    #         return base_price