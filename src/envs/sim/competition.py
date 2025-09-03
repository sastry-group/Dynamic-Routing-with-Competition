# competitor.py
from collections import defaultdict
from copy import deepcopy


class Allocator:
    """

    """
    def __init__(self, fleets, rule="equal", eps=1e-9):
        self.fleets = fleets
        self.rule = rule
        self.eps = eps
        self.t = 0
        self.firm_count = len(fleets)

    def compute_demand(self, demand_global_t):
        K = len(self.fleets)
        demand_per_firm = [defaultdict(float) for _ in range(K)]
        if self.rule == "equal":
            for e, D in demand_global_t.items():
                share = D / K
                for k in range(K):
                    demand_per_firm[k][e] = share
            return demand_per_firm


        # as a function of price 
        origins = {i for (i, _) in demand_global_t.keys()}
        weights = {i: [] for i in origins}
        for i in origins:
            accs = []
            for f in self.fleets:
                acc_i = f.acc[i].get(self.t+1, f.acc[i].get(self.t, 0.0))
                accs.append(max(0.0, acc_i))
            S = sum(accs) + self.eps
            weights[i] = [a / S for a in accs]

        for (i, j), D in demand_global_t.items():
            for k in range(K):
                demand_per_firm[k][(i, j)] = D * weights[i][k]
        return demand_per_firm

    def step(self):
        self.t += 1

    def compute_price(self, i, j, t, base_price, pricing_model):
        # print(pricing_model)
        # model: "cournot", "bertrand", "exogenous"
        if pricing_model == "cournot":
            try:
                q_total = sum(self.fleets[f].acc[i][t] for f in range(self.firm_count))
            except KeyError:
                q_total = sum(self.fleets[f].acc[i] for f in range(self.firm_count))
            if q_total <= 0:
                return base_price 

            # supply, number of initial vehicles (constant right now)
            # chekcing the q_total with total_Supplu
            a = base_price 
            alpha = 0.1
            b = alpha * a * (1 / q_total)
            cournot_price = a - b * q_total
            # print(supply, q_total, p) # or current planned quantity
            # print(f"Cournot price for edge ({i},{j}) at time {t}: {cournot_price}, and p,q: {p}, {q_total}")
            return cournot_price
        elif pricing_model == "bertrand":
            return base_price
        else:
            return base_price




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
    def __init__(self, scenario, fleets, allocator):
        self.scenario = scenario
        self.fleets = fleets
        self.allocator = allocator
        self.t = 0
        self.tf = scenario.tf

    def _market_publish(self):
        # shared total demand & price at time t
        D = {(i, j): self.scenario.demand_input[i, j].get(self.t, 0.0)
             for (i, j) in self.scenario.edges}
        P = {(i, j): self.scenario.p[i, j].get(self.t, 0.0)
             for (i, j) in self.scenario.edges}
        T_pax = {(i, j): self.scenario.demandTime[i, j].get(self.t, 0)
                 for (i, j) in self.scenario.edges}
        return D, P, T_pax

    def reset(self):
        self.t = 0
        self.allocator.t = 0
        # Re-init each fleet’s private state (use your existing reset logic)
        for f in self.fleets:
            # do a light reset that preserves scenario but resets fleet state
            f.t = 0
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
        tripAttr = self.scenario.get_random_demand(reset=True)
        self.regionDemand= defaultdict(dict)
        for i,j,t,d,p in tripAttr: # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i,j][t] = self.allocator.compute_demand(tripAttr)
            self.price[i,j][t] = self.allocator.compute_price(i,j,t,p,self.cfg.pricing_model)
            # self.demand[i,j][t] = d
            # self.price[i,j][t] = p
            # if t not in self.regionDemand[i]:
            #     self.regionDemand[i][t] = 0
            # else:
            #     self.regionDemand[i][t] +=d
            
        self.time = 0
        obs = []
        paxreward = []
        done = []
        info = []
        for f in self.fleets:
            for i,j in self.G.edges:
                f.rebFlow[i,j] = defaultdict(float)
                f.paxFlow[i,j] = defaultdict(float)            
            for n in self.G:
                f.acc[n][0] = self.G.nodes[n]['accInit']
                f.dacc[n] = defaultdict(float) 
            t = self.time
            for i,j in self.demand:
                f.servedDemand[i,j] = defaultdict(float)
            # TODO: define states here

            f.obs = (f.acc, f.time, f.dacc, f.demand)
            obs.append(f.obs)
            paxreward.append(f.paxreward)
            done.append(f.done)
            info.append(f.info)
            f.reward = 0
        return obs, paxreward


    def step(self, reb_actions_by_fleet):
        """
        reb_actions_by_fleet: list[dict[(i,j)->float]] from each model this step
        """
        # 1) market info
        D, P, T_pax = self._market_publish()

        # 2) deterministic demand split (no bidding)
        demand = self.allocator.compute_demand(D)

        # 3) each fleet solves its own pax LP with its cap + shared price
        #    Implemented as Fleet.match_with_caps(caps, P) returning {(i,j):flow}
        matched_list = []
        for f, caps in zip(self.fleets, demand):
            flows = f.match_with_caps(caps, P)  # your LP (adapted from matching_pulp)
            f.apply_pax(flows, T_pax, P)        # updates revenue/costs/private state
            matched_list.append(flows)

        # 4) apply private rebalancing from the agents
        for f, reb in zip(self.fleets, reb_actions_by_fleet):
            f.rebalancing(reb)

        # 5) arrivals + time advance
        for f in self.fleets:
            f.advance()
        self.t += 1
        self.allocator.step()

        done = (self.t >= self.tf)
        # collect per-fleet per-step info if needed
        infos = [f.info.copy() for f in self.fleets]
        return done, infos