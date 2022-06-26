from pydantic import BaseModel
from typing import List, Optional, Union, Dict

class Technology(BaseModel):
    pass

class GenerationTechnology(Technology):
    name: str
    construction_lead_years : int = 1 # years
    emissions_rate: float             # t / MW-year
    area_footprint: float             # sqkm / MW
    capital_cost_var: float           # $/kW
    capital_cost_fixed: float         # $
    capital_cost_gr: float            # e.g. -0.05 -> 5% decrease/yr
    opex: float                       # $ / kW
    lifespan: int                     # number of years
    
    
class TransmissionTechnology(Technology):
    name: str
    construction_lead_years : int = 1  # years
    loss: float                        # % / km
    capital_cost_var: float            # $/KW/km
    capital_cost_fixed: float          # $
    capital_cost_gr: float = 0.        # e.g. 0.06 -> 6% increase/yr
    lifespan: int                      # number of years
    
class InitialTech(Technology):
    ob_type: str
    ob_key: Union[str,tuple]
    tech_type: str
    capacity: float # kw
    age: int        # years
    
class Land(BaseModel):
    name: str
    cost: float                        # $ / sqm

class Node(BaseModel):
    id: str
    solar_irradiance: float            # kWh/kW
    area: float                        # sqkm
    initial_landarea: dict              # dict(lc_key:percentage)
    initial_population: int             # people
    node_technologies: Dict[str,GenerationTechnology]
    
class Edge(BaseModel):
    id: tuple
    distance: float                    # km
    edge_technologies: Dict[str,TransmissionTechnology]
    
class GrowthRate(BaseModel):
    max_rate: float
    min_rate:float 
    peak_year:int
    half_decay:int
    
class GridlandiaGlobals(BaseModel):
    population_gr: GrowthRate
    final_energy_intensity: int # kWh/person/yr
    final_energy_intensity_gr: GrowthRate 
    
class GridLandiaConstraints(BaseModel):
    net_zero_year: Optional[int]
    land_constraints: Optional[int]