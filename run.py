
import sys, random
from QJJ_model_1 import optz_model_1
from QJJ_model_2 import optz_model_2
from QJJ_model_3 import optz_model_3, getResult
from QJJ_model_4 import optz_model_4, run_model_4_rounds
from part_raw_data import *
from print_results import *

details = True

def run_test(seeds=None, reGenData=False, rounds=1):

    default = reGenData
    setRegenDefault(default)
    if seeds is not None:
        global_seed, carbon_emissions_seed = seeds
    else:
        global_seed, carbon_emissions_seed = 1, 2
    random.seed(global_seed)

    data_path = 'data/'

    raw_data_path = data_path + '/raw'
    parts = data_path + 'fixed_data/all_parts.xlsx'
    locations = data_path + 'fixed_data/locations.xlsx'
    costs = data_path + 'costs.xlsx'
    schedules = data_path + 'schedules.xlsx'
    demands = data_path + 'demands.xlsx'
    availabilities = data_path + 'availabilities.xlsx'
    capacities = data_path + 'capacities.xlsx'
    carbon = data_path + 'carbon.xlsx'

    VehicleSpeed = 100 # per hour
    
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    all_ports, all_plants, all_dcs = load_locations(locations)
    v_records, v_vessels, v_ports = load_valid_schedules(schedules, path=raw_data_path, ports=all_ports, thd1=10, regen=False)
    all_parts, all_insps, all_insp_pts = load_parts_inspections(parts, pl=15)
    port_demands = load_port_demands(demands, vports=v_ports, vparts=all_parts)
    part_plant_avails = load_part_at_plant_availibility(availabilities, vplants=all_plants, vparts=all_parts)
    pl_caps = load_plant_capacity(capacities, vplants=all_plants, vparts=all_parts, vport_demands=port_demands, vsupply=part_plant_avails)
    all_distances = load_valid_distances(locations, vplants=all_plants, vdcs=all_dcs, vports=v_ports)
    v_trans_times = {k: int(v / VehicleSpeed) for k,v in all_distances.items()}
    trans_costs = load_trans_cost(costs, vdistances=all_distances, vparts=all_parts)
    vessel_demands = load_vessel_demands(demands, vessels=v_vessels, vparts=all_parts, 
                                         vplants=all_plants, vsupply=part_plant_avails, vports=v_ports, 
                                         vtimes=v_trans_times, vschedules=v_records)
    holding_costs = load_holding_costs(costs, vports=v_ports, vparts=all_parts)
    part_plant_shipout = load_plant_shipout_cost(costs, vlines=all_distances.keys())
    insp_port_avails = load_insp_at_port_availability(availabilities, vports=v_ports, vinsps=all_insps)
    insp_costs = load_inspection_cost(costs, vports=v_ports, vinsps=all_insps)
    vessel_insp_demands = load_vessel_insp_demands(demands, vessels=v_vessels, vinsps=all_insps, vdemand=vessel_demands, 
                                                   vschedule=v_records, insp_avail=insp_port_avails, 
                                                   vplants=all_plants, vsupply=part_plant_avails, 
                                                   vtimes=v_trans_times, vinsps_parts=all_insp_pts)
    dc_caps, dc_costs = load_dc_capacity_and_setup_cost(capacities, vdcs=all_dcs, vport_demands=port_demands)
    

    random.seed(carbon_emissions_seed)
    carbon_emissions = load_carbon_emmissions(carbon, vdistances=all_distances, vparts=all_parts)
    carbonRate = 0.8


    def run_model_1():
        return optz_model_1(all_parts, all_plants, pl_caps, all_dcs, dc_caps, dc_costs, v_ports, port_demands, trans_costs, all_distances, details)
        
    def run_model_2():
        return optz_model_2(all_parts, all_plants, v_vessels, v_ports, part_plant_avails, trans_costs, part_plant_shipout, 
                    all_distances, v_records, vessel_demands, holding_costs, VehicleSpeed, details)

    def run_model_3():
        return optz_model_3(all_parts, all_insps, all_plants, v_vessels, v_ports, part_plant_avails, insp_port_avails, 
                    trans_costs, all_insp_pts, part_plant_shipout, all_distances, v_records, vessel_demands, 
                    vessel_insp_demands, holding_costs, insp_costs, VehicleSpeed, details)

    def run_model_4():
        return optz_model_4(all_parts, all_insps, all_plants, v_vessels, v_ports, part_plant_avails, insp_port_avails, 
                    trans_costs, all_insp_pts, part_plant_shipout, all_distances, v_records, vessel_demands, 
                    vessel_insp_demands, holding_costs, insp_costs, carbon_emissions, VehicleSpeed, 0, details)
    
    def run_model_4_s():
        v_vessels_s = v_vessels[:2]
        print(f'  run on the vessels {v_vessels_s}')
        vessel_demands_s = {k:v for k,v in vessel_demands.items() if k in v_vessels_s}
        vessel_insp_demands_s = {k:v for k,v in vessel_insp_demands.items() if k in v_vessels_s}
        # v_vessels_s = v_vessels
        # vessel_demands_s = vessel_demands
        # vessel_insp_demands_s = vessel_insp_demands
        if getResult() is None:
            print_line(f'Run Model 3', sp='+')
            optz_model_3(all_parts, all_insps, all_plants, v_vessels_s, v_ports, part_plant_avails, insp_port_avails, 
                    trans_costs, all_insp_pts, part_plant_shipout, all_distances, v_records, vessel_demands_s, 
                    vessel_insp_demands_s, holding_costs, insp_costs, VehicleSpeed, False)
            print_line(f'End Model 3', sp='+')
            print('\n\n\n\n')
        minBudget = getResult()[0]['Value'][0]
        return run_model_4_rounds(all_parts, all_insps, all_plants, v_vessels_s, v_ports, part_plant_avails, insp_port_avails, 
                    trans_costs, all_insp_pts, part_plant_shipout, all_distances, v_records, vessel_demands_s, 
                    vessel_insp_demands_s, holding_costs, insp_costs, carbon_emissions, VehicleSpeed, minBudget, 0, rounds, details)

    models = {'1': run_model_1, '2': run_model_2, '3': run_model_3, '4': run_model_4, '4s': run_model_4_s}
    
    if len(sys.argv) == 1:
        argv = ['1', '2', '3', '4', '4s']
    else:
        argv = sys.argv[1:]
    argv = ['4s']
    res = 0
    for i in argv:
        print_line(f'Run Model {i}', sp='+')
        res += models[i]()
        print_line(f'End Model {i}', sp='+')
        print('\n\n\n\n')
    return int(res == len(argv))

def find_random_seeds():
    i = random.randint(0, 1000000)
    while True:
        random.seed(i)
        i += 1
        gs = random.randint(0, 10000000)
        cs = random.randint(0, 10000000)
        f = run_test((gs,cs), True)
        if f == 1:
            print('global_seed: ', gs, ', carbon_emissions_seed: ', cs)
            break

if __name__ == '__main__':
    # find_random_seeds()
    # run_test((4916192, 7854691), True)
    # run_test(reGenData=False)
    run_test(rounds=10)
