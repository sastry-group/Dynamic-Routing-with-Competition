import importlib
import testing
import warnings
warnings.filterwarnings("ignore")
importlib.reload(testing)


import sys
if 'testing' in sys.modules:
    del sys.modules['testing']

import testing
importlib.reload(testing)
# config = {
#     "simulator.name": "multi_macro",
#     "model.name": ["sac", "equal_distribution", "random"],
#     "simulator.city": "nyc_brooklyn", 
#     "model.cplexpath": None, 
#     "model.test_episodes": 2,
#     "model.checkpoint_path": ["SAC_portion_2","SAC_portion_2"],
#     "simulator.reuse_no_control": False,
#     "simulator.firm_count": 2,
#     "simulator.agents_know_partial_demand": True,
#     "simulator.constant_vehicle_count": True,
#     "simulator.demand_filter_type": "competition" ,
#     "simulator.pricing_model": "cournot"

# }

# config = {
#     "simulator.name": "multi_macro",
#     "model.name": ["sac", "equal_distribution", "random"],
#     # "model.name": ["sac"],
#     "simulator.city": "nyc_brooklyn",
#     # "simulator.demand": "historical_demand_firm_0_0",
#     "model.cplexpath": None,
#     "model.test_episodes": 1,
#     "model.checkpoint_path": "SAC_flow_4",
#     "model.loop_number": 0,
#     "simulator.reuse_no_control": False,
#     "simulator.firm_count": 4,
#     "simulator.agents_know_partial_demand": True,
#     "simulator.constant_vehicle_count": True,
#     "simulator.demand_filter_type": "equal" ,
#     "simulator.pricing_model": "equal"
# }



config = {
    "simulator.name": "multi_macro",
    "model.name": ["sac", "equal_distribution", "random"],
    "simulator.city": "san_francisco", 
    "simulator.demand": "san_francisco",
    "model.cplexpath": None, 
    "model.test_episodes": 10,
    "model.checkpoint_path": "SAC_testNOM_firm1_SF",
    "simulator.reuse_no_control": True,
    "simulator.firm_count": 1,
    "simulator.agents_know_partial_demand": False,
    "simulator.alpha": 0.8,
    "model.loop_number": 0,
    "simulator.competition": False,
    "simulator.constant_vehicle_count": True,
    "simulator.demand_filter_type": "flow" ,
    "simulator.pricing_model": "exogenous"

    }

testing.multi_test(config)

