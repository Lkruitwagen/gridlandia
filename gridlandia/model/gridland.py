from typing import List, Optional

import geopandas as gpd
from geopy import distance
from pulp import *

from itertools import product

from gridlandia.model import schemas
from gridlandia.model.utils import *


class GridLand(object):
    """geometry definitions etc."""
    
    
    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        landcover_classes: List[str],
        column_map: Optional[dict]=None,
    ):
        # check CRS of gdf, warn if None
        # TODO
        
        # assert that gdf columns exist
        if column_map is not None:
            for key in ['node_id','area','population','solar_irradiance']+[c+'_area' for c in landcover_classes]:
                assert key in column_map.values(), f'{key} not in {column_map.values()}'
            
            gdf = gdf.rename(columns=column_map)
        
        
        self.grid_gdf = gdf
        
        self.node_ids = gdf['node_id'].unique()
        
        # dissolve into nodes for MILP
        aggfuncs = {col:'sum' for col in gdf.columns if 'area' in col or col=='population'}
        aggfuncs.update({'solar_irradiance':'mean'})
        self.gdf = gdf.dissolve('node_id', aggfunc=aggfuncs)
        print(self.gdf)
        
        # spatial join to get neighbours
        self.edges = set()
        for e in gpd.sjoin(self.gdf,self.gdf).reset_index()[['node_id','index_right']].values.tolist():
            if e[0]!=e[1] and tuple(sorted(e)) not in self.edges:
                self.edges.add(tuple(sorted(e)))
        
        self.distances = {}
        for e in self.edges:
            dist = distance.distance(self.gdf.loc[e[0],'geometry'].centroid.coords[0][::-1],self.gdf.loc[e[1],'geometry'].centroid.coords[0][::-1]).km
            self.distances[e] = dist
            self.distances[(e[1],e[0])] = dist
        
        
        
class GridLandModel(object):
    
    def __init__(
        self,
        gridland: GridLand,
        years: List[int],
        global_params: schemas.GridlandiaGlobals,
        global_constraints: schemas.GridLandiaConstraints,
        node_technology_defaults: dict,
        edge_technology_defaults: dict,
        initial_technologies: Optional[List[schemas.InitialTech]]=None,
        node_technology_overwrites: Optional[dict] = None,
        edge_technology_overwrites: Optional[dict] = None,
    ):
        
        
        # #### SETUP - TECHS #####
        generation_techs = {kk:schemas.GenerationTechnology(name=kk,**vv) for kk,vv in node_technology_defaults.items()}
        transmission_techs = {kk:schemas.TransmissionTechnology(name=kk,**vv) for kk,vv in edge_technology_defaults.items()}
        
        
        
        # #### SETUP -  NODES & EDGES #####
        nodes = {}
        for idx, row in gridland.gdf.iterrows():
            nodes[idx] = schemas.Node(
                id=idx,
                solar_irradiance=row['solar_irradiance'],
                area=row["area"],
                initial_landarea=row[[col for col in row.index if '_area' in col]].to_dict(),
                initial_population=row["population"],
                node_technologies=generation_techs
            )
            
        edges = {
            e:schemas.Edge(
                id=e, 
                distance=gridland.distances[e],
                edge_technologies=transmission_techs
            )
            for e in gridland.edges
        }
        
        # #### SETUP -  INITIAL CAPACITIES #####
        initial_node_capacity = fill_product({}, nodes.keys(), generation_techs.keys(),years, 0)
        initial_edge_capacity = fill_product({}, edges.keys(), transmission_techs.keys(),years, 0)

        if initial_technologies is not None:
            for initial_tech in initial_technologies:
                if initial_tech.ob_type=='node':
                    for year in range(min(years),min(years)+generation_techs[initial_tech.tech_type].lifespan-initial_tech.age):
                        initial_node_capacity[initial_tech.ob_key][initial_tech.tech_type][year] += initial_tech.capacity
                elif initial_tech.ob_type=='edge':
                    for year in range(min(years),min(years)+transmission_techs[initial_tech.tech_type].lifespan-initial_tech.age):
                        initial_edge_capacity[initial_tech.ob_key][initial_tech.tech_type][year] += initial_tech.capacity
                    
                    
        # #### SETUP -  NODE DEMAND #####
        population_gr_years = growth_rate_bell(**global_params.population_gr.__dict__)(np.array(years))
        energy_intensity_gr_years = growth_rate_bell(**global_params.final_energy_intensity_gr.__dict__)(np.array(years))

        node_demand = fill_product({}, nodes.keys(), years,None,0)
        for node_id in nodes.keys():
            for ii_y, year in enumerate(years):
                # node_demand = pop_in_node_yr * energy_intensity_in_node_yr / 8760
                pop_in_node_yr = nodes[node_id].initial_population*np.prod((1.+population_gr_years[:ii_y]))
                energy_intensity_in_node_yr = global_params.final_energy_intensity*np.prod((1.+energy_intensity_gr_years[:ii_y]))
                node_demand[node_id][year] = pop_in_node_yr * energy_intensity_in_node_yr / 8760
                
                
        self.build_model(
            years,
            node_demand,
            global_constraints,
            gridland.distances,
            generation_techs,
            transmission_techs,
            nodes,
            edges,
            initial_node_capacity,
            initial_edge_capacity
        )
        
        
        
    def build_model(
        self,
        years,
        node_demand,
        global_constraints,
        all_distances,
        generation_techs,
        transmission_techs,
        nodes,
        edges,
        initial_node_capacity,
        initial_edge_capacity,
    ):
        
        reverse_edge_keys = [(e[1],e[0]) for e in edges.keys()]
        
        
        # ######## INITIALISE MODEL ######
        model = LpProblem("EnergyCapacityLeastCosts",LpMinimize)
        
        
        # ######## SOLUTION VARS ##########
        
        F = LpVariable.dicts("Flows",(reverse_edge_keys+list(edges.keys()),transmission_techs.keys(),years), lowBound=0, cat='continuous')
        # capacities_additions in [nodes, technologies, year]
        cap_add_nodes = LpVariable.dicts("CapAddNodes", (nodes.keys(), generation_techs.keys(), years), lowBound=0, cat='continuous')
        # capacity_additions in [edges, technologies, year]
        cap_add_edges = LpVariable.dicts("CapAddEdges", (edges.keys(), transmission_techs.keys(), years), lowBound=0, cat='continuous')
        # Generation in [nodes, technologies, year]
        gen_nodes = LpVariable.dicts("Gen", (nodes.keys(), generation_techs.keys(), years), lowBound=0, cat='continuous')
        
        # transit vars (boolean forcing) 
        # https://cs.stackexchange.com/questions/69531/greater-than-condition-in-integer-linear-program-with-a-binary-variable
        construction_node_bool = LpVariable.dicts("ConstrNodes", (nodes.keys(), generation_techs.keys(), years), 0,1, LpInteger)
        construction_edge_bool = LpVariable.dicts("ConstrEdges", (edges.keys(), transmission_techs.keys(), years), 0,1, LpInteger)
        flow_direction_bool = LpVariable.dicts("FlowDirection", (edges.keys(), transmission_techs.keys(), years), 0,1, LpInteger)
        

        # ######## AFFINE EQUATIONS #########
                               
        # build affines
        supply_capacities = fill_product({}, nodes.keys(), generation_techs.keys(),years, 0)
        for n, t, y in product(nodes.keys(),generation_techs.keys(),years):
            # initial + additional - retirement
            supply_capacities[n][t][y] = initial_node_capacity[n][t][y] + sum([cap_add_nodes[n][t][build_year] for build_year in range(max(min(years),y-generation_techs[t].lifespan),y+1)])

        transmission_capacities = fill_product({}, edges.keys(), transmission_techs.keys(),years, 0)
        for e, t, y in product(edges.keys(),transmission_techs.keys(),years):
            # initial + additional - retirement
            transmission_capacities[e][t][y] = initial_edge_capacity[e][t][y] + sum([cap_add_edges[e][t][build_year] for build_year in range(max(min(years),y-transmission_techs[t].lifespan),y+1)])

        supply = fill_product({}, nodes.keys(), years, None, 0)
        for n, y in product(nodes.keys(), years):
            supply[n][y] = sum([gen_nodes[n][t][y] for t in generation_techs.keys()])

        netflow = fill_product({}, nodes.keys(), years, None, 0)
        for n, y in product(nodes.keys(), years):
            # sum( inflow*eff - outflow)
            netflow[n][y] = sum([F[e][t][y]*(1-transmission_techs[t].loss*all_distances[e])  for e in reverse_edge_keys+list(edges.keys()) for t in transmission_techs.keys() if e[1]==n]) - \
                                     sum([F[e][t][y] for e in reverse_edge_keys+list(edges.keys()) for t in transmission_techs.keys() if e[0]==n])
    
        emissions = {}
        for y in years:
            emissions[y] = sum([gen_nodes[n][t][y]*generation_techs[t].emissions_rate for n in nodes.keys() for t in generation_techs.keys()])
            
            
        # ####### CONSTRAINTS - MIN-COST-FLOW PROBLEM ###########
        
        # node demand satisficing and conservation of energy -> supply in techs + netflow - demand == 0 in (nodes, years)
        for n, y in product(nodes.keys(), years):
            model += supply[n][y] + netflow[n][y] - node_demand[n][y] == 0

        # Supply availability -> total supply >= total demand
        for y in years:
            model += sum([supply[n][y] for node_id in nodes.keys()]) >= sum([node_demand[n][y] for n in nodes.keys()])
            
        # supply capacity -> node generation <= node supply capacity
        for n, t, y in product(nodes.keys(), generation_techs.keys(), years):
            model += gen_nodes[n][t][y]<=supply_capacities[n][t][y]
    
        # edge flow capacity -> edge transmission <= edge transmission capacity
        for e, t, y in product(edges.keys(), transmission_techs.keys(), years):
            model += F[e][t][y]<=transmission_capacities[e][t][y]
            # constrain the reverse direction also
            model += F[(e[1],e[0])][t][y]<=transmission_capacities[e][t][y]
            
            
        # ####### CONSTRAINTS - BOOLEAN FORCING ###########
            
        M = max([max(d.values()) for d in list(node_demand.values())])*1000 # a big number
        
        # construction period booleans
        for n, t, y in product(nodes.keys(), generation_techs.keys(), years):
            model += cap_add_nodes[n][t][y] >= 1+-M*(1-construction_node_bool[n][t][y]) # B≥C+1−M(1−A)
            model += cap_add_nodes[n][t][y] <= M*construction_node_bool[n][t][y] # B≤C+MA

        for e, t, y in product(edges.keys(), transmission_techs.keys(), years):
            model += cap_add_edges[e][t][y] >= 1+-M*(1-construction_edge_bool[e][t][y]) # B≥C+1−M(1−A)
            model += cap_add_edges[e][t][y] <= M*construction_edge_bool[e][t][y] # B≤C+MA
        
        # edge flow direction can only go one way per timeperiod
        for e, t, y in product(edges.keys(), transmission_techs.keys(), years):
            model += F[e][t][y] <= M*flow_direction_bool[e][t][y]
            model += F[(e[1],e[0])][t][y] <= M*(1-flow_direction_bool[e][t][y])
            
        # ####### CONSTRAINTS - SCENARIO-DRIVEN ###########
        
        # constrain emissions
        if global_constraints.net_zero_year is not None:
            for y in years:
                if y>=global_constraints.net_zero_year:
                    model+= emissions[y] == 0.
                
        # ######## TARGET - TOTAL COST ##############
        
        # cost-years
        # simply for now, just fixed + variable in cap add
        cost = {}
        for y in years:
            cost[y] = sum([cap_add_nodes[n][gt][y]*generation_techs[gt].capital_cost_var for n,gt in product(nodes.keys(), generation_techs.keys())]) + \
                         sum([cap_add_edges[e][tt][y]*transmission_techs[tt].capital_cost_var for e,tt in product(edges.keys(), transmission_techs.keys())]) + \
                         sum([construction_node_bool[n][gt][y]*generation_techs[gt].capital_cost_fixed for n,gt in product(nodes.keys(), generation_techs.keys())]) + \
                         sum([construction_edge_bool[e][tt][y]*transmission_techs[tt].capital_cost_fixed for e,tt in product(edges.keys(), transmission_techs.keys())]) + \
                         sum([gen_nodes[n][gt][y]*generation_techs[gt].opex for n,gt in product(nodes.keys(),generation_techs.keys())])
            
        total_cost = sum([cost[y] for y in years])
        
        model += total_cost, "Total Cost"
                
        self.model = model
        self.cost = cost
        self.total_cost = total_cost
        self.F = F
        self.cap_add_nodes = cap_add_nodes
        self.cap_add_edges = cap_add_edges 
        self.gen_nodes = gen_nodes
        self.construction_node_bool = construction_node_bool
        self.construction_edge_bool = construction_edge_bool
        self.flow_direction_bool = flow_direction_bool
        self.supply_capacities = supply_capacities
        self.transmission_capacities = transmission_capacities
        self.supply = supply
        self.netflow = netflow
        self.emissions = emissions

        
    def solve(self):
        self.model.solve(GUROBI_CMD())
        
    def display_solution(self):
        
        if LpStatus[self.model.status] != 'Optimal':
            raise ValueError('Model status not Optimal')
            
        # TODO: get this into some tables
            
        print ("#### COST ####")
        for kk,vv in self.cost.items():
            print (kk,vv.value())

        print ("#### NET FLOW ####")
        for kk,vv in self.netflow.items():
            for kk2, vv2 in vv.items():
                print (kk, kk2, f'{vv2.value():.3f}')

        print ("#### FLOW ####")
        for kk,vv in self.F.items():
            for kk2, vv2 in vv.items():
                for kk3, vv3 in vv2.items():
                    print (kk,kk2,kk3,f'{vv3.value():.3f}')

        
        print ("#### GEN NODES ####")
        for kk,vv in self.gen_nodes.items():
            for kk2, vv2 in vv.items():
                for kk3, vv3 in vv2.items():
                    print (kk,kk2,kk3,f'{vv3.value():.3f}')


        print ("#### CAP ADD - NODES ####")
        for kk,vv in self.cap_add_nodes.items():
            for kk2, vv2 in vv.items():
                for kk3, vv3 in vv2.items():
                    print (kk,kk2,kk3,f'{vv3.value():.3f}')

        print ("#### CAP ADD - EDGES ####")
        for kk,vv in self.cap_add_edges.items():
            for kk2, vv2 in vv.items():
                for kk3, vv3 in vv2.items():
                    print (kk,kk2,kk3,f'{vv3.value():.3f}')

        print ("#### CONSTRUCTION NODE BOOL ####")
        for kk,vv in self.construction_node_bool.items():
            for kk2, vv2 in vv.items():
                for kk3, vv3 in vv2.items():
                    print (kk,kk2,kk3,f'{vv3.value():.3f}')

            
        
        
        