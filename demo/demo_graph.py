from pydantic import BaseModel
from itertools import chain, product
from typing import Optional

import networkx as nx
from pulp import *

from loguru import logger

class Plant(BaseModel):
    id:str                       # id
    remaining_life: int          # years
    output_capacity: int                # kW
    io_ratio: float              # efficiency
    capex: int                   # $
    opex: int                    # $/KWh
    commodity_input: str         # INPUT commodity
    commodity_output: str        # OUTPUT commodity
    availability_year: int       # first year
    inp_nodes: Optional[list] = None
    outp_nodes: Optional[list] = None
    capexp_node: Optional[str] = None
    active_yrtimes: Optional[list] = None
    input_capacity: int = None
    
    
def build_graph(all_plants, commodities, times, yr_time_demand):
    
    G = nx.DiGraph()
    
    # commodity buses
    commodity_nodes = ['-'.join([commodity,yr_time]) for commodity, yr_time in product(commodities, yr_time_demand.keys())]
    
    G.add_nodes_from(commodity_nodes)
    
    # add existing plants and new plants
    for plant in all_plants:
        # add input nodes
        plant.active_yrtimes = [[yr,t] for yr, t in product(range(plant.availability_year, min(30,plant.availability_year+plant.remaining_life)), times)]
        plant_nodes = [f'{plant.id}-{yr}-{t}' for yr, t in product(range(plant.availability_year, min(30,plant.availability_year+plant.remaining_life)), times)]
        G.add_nodes_from(plant_nodes)
        
    plants_dict = {p.id: p for p in all_plants}
    
    # add optimisation nodes
    global_demand = sum(yr_time_demand.values())
    LARGE_VAL = global_demand*10
    optim_nodes = [
        ('GLOBAL_DEMAND',{'demand':global_demand}),
        ('GLOBAL_SUPPLY',{'demand':-LARGE_VAL}),
        ('LOSS_SINK',{'demand':LARGE_VAL-global_demand}),
    ]
    G.add_nodes_from(optim_nodes)
    
    ### ENERGY FLOW
    # input & output edges
    input_edges = [ ('GLOBAL_SUPPLY','-'.join(['INPUT',yr_time]),{'capacity':LARGE_VAL,'cost_var':0,'cost_fixed':0}) for yr_time in yr_time_demand.keys()]
    output_edges = [('-'.join(['OUTPUT',yr_time]),'GLOBAL_DEMAND',{'capacity':val, 'cost_var':0, 'cost_fixed':0}) for yr_time, val in yr_time_demand.items()]
    G.add_edges_from(input_edges+output_edges)
    
# plant endges
    for plant in all_plants:
        input_edges =  [('-'.join([plant.commodity_input,str(yr),str(t)]), f'{plant.id}-{yr}-{t}', {'capacity':plant.input_capacity, 'cost_var':0, 'cost_fixed':0}) for yr,t in plant.active_yrtimes]
        output_edges = [(f'{plant.id}-{yr}-{t}','-'.join([plant.commodity_output,str(yr),str(t)]), {'capacity':plant.output_capacity, 'cost_var':plant.opex, 'cost_fixed':0}) for yr,t in plant.active_yrtimes]
        loss_edges = [(f'{plant.id}-{yr}-{t}','LOSS_SINK', {'capacity':plant.input_capacity-plant.output_capacity, 'cost_var':0, 'cost_fixed':0}) for yr,t in plant.active_yrtimes]
        
        G.add_edges_from(input_edges+output_edges+loss_edges)
        
    # shortcut edge
    G.add_edges_from([('GLOBAL_SUPPLY','LOSS_SINK',{'capacity':LARGE_VAL, 'cost_var':0,'cost_fixed':0})])
        
    return G
    
    
def build_model(G, plants_dict):
    
    model = LpProblem("graph_model_mincostflow",LpMinimize)
    
    # variables: only 2; 
    F = LpVariable.dicts("Flows",list(G.edges), lowBound=0, cat='continuous')
    capexp = LpVariable.dicts("CapExp", list(plants_dict.keys()), 0,1, LpInteger)

    ### CONSTRAINTS ###
    logger.info('ADDING CONSTRAINTS')
    # for every node, sum(flows) + demand == 0
    for n, params in G.nodes.items():
        model += sum([F[e_i] for e_i in G.in_edges(n)])-sum([F[e_o] for e_o in G.out_edges(n)]) - params.get('demand',0) == 0


    
    # for every edge, constrain capacity
    for (n_s, n_t), data in G.edges.items():
        if 'plant' in n_s:
            p_id = '-'.join(n_s.split('-')[0:4])
            #print (p_id)
            model += F[(n_s, n_t)] <= data['capacity'] * capexp[p_id]
        elif 'plant' in n_t:
            p_id = '-'.join(n_t.split('-')[0:4])
            #print (p_id)
            model += F[(n_s, n_t)] <= data['capacity'] * capexp[p_id]
        else:
            model += F[(n_s, n_t)] <= data['capacity']

    
    # for each plant node, the LOSS_SINK flow == sum(flow_inp) * io_ratio
    for n_s, data in G.nodes.items():
        if 'plant' in n_s:
            p_id = '-'.join(n_s.split('-')[0:4])
            model += F[(n_s, 'LOSS_SINK')] == sum([F[e_i] for e_i in G.in_edges(n_s)]) * (1. - plants_dict[p_id].io_ratio)

    

    # sum costs
    logger.info('ADDING OPTIM CRITERIA')
    flow_costs = [F[(n_s, n_t)] * e_data['cost_var'] for (n_s, n_t), e_data in G.edges.items()]
    capexp_costs = [capexp[p_id] * plants_dict[p_id].capex for p_id in capexp.keys()]
    total_cost = sum(flow_costs) + sum(capexp_costs)
    model += total_cost, "Total Cost"
    
    return model, F, capexp

class graphModel:
    
    def __init__(self, G, plants_dict):
        
        self.model, self.F, self.capexp = build_model(G, plants_dict)
        
    def solve(self):
        
        self.model.solve(GUROBI_CMD())
        
    def report(self):
        
        print(LpStatus[self.model.status])

        print ('FLOWS:')
        for kk,vv in self.F.items():
            if vv.value()>0:
                print (kk,vv.value())
                
        print ('BUILT PLANTS:')
        for kk,vv in self.capexp.items():
            if vv.value()>0:
                print (kk,vv.value())
        
        
    
        
    
def graph_demo():
    
    logger.info('START')
    
    initial_plants = [
        Plant(
            id='plant-existing-0-0',
            remaining_life=5,
            output_capacity = 2000,
            io_ratio = 0.8,
            capex = 0,
            opex = 1,
            commodity_input = "INPUT",
            commodity_output = "OUTPUT",
            availability_year = 0,
            input_capacity = 2000/0.8,
        ),
        Plant(
            id='plant-existing-0-1',
            remaining_life=15,
            output_capacity = 5000,
            io_ratio = 0.6,
            capex = 0,
            opex=2,
            commodity_input = "INPUT",
            commodity_output = "OUTPUT",
            availability_year = 0,
            input_capacity = 5000/0.6,
        )
    ]
    
    new_plants = [
        [
            Plant(
                id=f'plant-new-{year}-0',
                remaining_life=10,
                output_capacity=1000,
                io_ratio = 0.8-0.2*(year/30),
                capex = 1500*1000,
                opex = 1,
                availability_year = year,
                commodity_input = "INPUT",
                commodity_output = "OUTPUT",
                input_capacity = 1000/( 0.8-0.2*(year/30)),
            ),
            Plant(
                id=f'plant-new-{year}-1',
                remaining_life=20,
                output_capacity=2000,
                io_ratio =  0.6-0.1*(year/30),
                capex = 1000*2000,
                opex = 2,
                availability_year = year,
                commodity_input = "INPUT",
                commodity_output = "OUTPUT",
                input_capacity = 2000/(0.6-0.1*(year/30)),
            ),
        ]
        for year in range(30)
    ]
    all_plants = initial_plants + list(chain(*new_plants))
    
    times = range(12)
    years = range(30)
    peak_demands = {yr:6000+500*yr for yr in years}
    yr_time_demand = {f'{yr}-{t}':0.5*peak_demands[yr]+0.5*peak_demands[yr]*(1-(abs((t-6)/6))) for yr, t in product(years, times)}
    commodities = ["INPUT","OUTPUT"]
    
    logger.info('BUILDING GRAPH')
    G = build_graph(all_plants, commodities, times, yr_time_demand)
    plants_dict = {p.id: p for p in all_plants}
    
    logger.info('BUILDING MODEL')
    model = graphModel(G, plants_dict)
    
    logger.info('SOLVING MODEL')
    model.solve()
    
    logger.info('REPORT')
    model.report()
    
    
    
if __name__=="__main__":
    graph_demo()