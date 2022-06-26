import geopandas as gpd
import yaml

from gridlandia.model import schemas
from gridlandia.model import GridLand, GridLandModel
from gridlandia.model.defaults import *

def main(gdf_path, params_path):
    gdf = gpd.read_file(gdf_path)
    params = yaml.load(open(params_path,'r'), Loader=yaml.SafeLoader)
    
    gridland = GridLand(
        gdf=gdf,
        landcover_classes=["builtup","plantation","barren","forest","agriculture"],
        column_map=None,
    )
    
    years = list(range(params['start_year'], params['end_year']+1))
    gridlandia_globals = schemas.GridlandiaGlobals(
        population_gr= schemas.GrowthRate(**params['population_gr']),
        final_energy_intensity= params['final_energy_intensity'],
        final_energy_intensity_gr = schemas.GrowthRate(**params['final_energy_intensity_gr']),
    )
    gridlandia_constraints = schemas.GridLandiaConstraints(
        net_zero_year=params['net_zero_year']
    )
    initial_technologies = [
        schemas.InitialTech(
            ob_type=tech['ob_type'],
            ob_key=tech['ob_key'] if isinstance(tech['ob_key'],str) else tuple(tech['ob_key']),
            tech_type=tech['tech_type'],
            capacity=tech['capacity'],
            age=tech['age'], 
        )
        for tech in params['initial_technologies']]
    
    
    model = GridLandModel(
        gridland = gridland,
        years = years,
        global_params=gridlandia_globals,
        global_constraints = gridlandia_constraints,
        node_technology_defaults = generation_technologies,
        edge_technology_defaults = transmission_technologies,
        initial_technologies = initial_technologies,
        node_technology_overwrites = None,
        edge_technology_overwrites = None,
    )
    
    model.solve()
    
    model.display_solution()
    
if __name__=="__main__":
    main(
    gdf_path='./demo/toy_2node_gridland.geojson',
    params_path='./demo/toy_2node_params.yaml',
    )