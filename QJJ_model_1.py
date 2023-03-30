# -*- coding: utf-8 -*-
""" Transportation problem Winston Example """
# Updated 2022 09 19
import gurobipy as gp
from gurobipy import GRB
from print_results import *
from part_raw_data import write2Excel


def optz_model_1(Products, Plants, Supply, DCs, 
                 DCCaps, DCCosts , Destinations, 
                 Demand, TransCost, Distances, details=True):
            
    total_demand = sum([v for k,v in Demand.items()])
    Vars_distances = [(i,j) for i,j in Distances.keys() if i in Plants or i in DCs and j in DCs or j in Destinations]
    # [print(v) for v in Vars_distances]
    # print(len(Vars_distances), len(set(Vars_distances)))
    ## Model
    m = gp.Model("Problem_1")

    ## Decision Variables
    # x[(i,d,k)] = quantity shipped from plant i to DC d for product k or from DC d to destination j for product k 
    x = m.addVars(Vars_distances, Products, vtype=GRB.INTEGER, name="x")
    
    # y[d] = whether DC d is estabilished
    y = m.addVars(DCs, vtype=GRB.BINARY, name="y")
    


    # Objective function
    # Minimize the total cost
    m.setObjective(
        gp.quicksum(x[i,j,p]*TransCost[i,j,p]*Distances[i,j] for i,j in Vars_distances for p in Products) + \
        gp.quicksum(y[d]*DCCosts[d] for d in DCs),
            GRB.MINIMIZE)

    ## Constraints
    # Supplies from each plant i and product k are limited by its supply capacity
    c1 = m.addConstrs(
        (gp.quicksum(x[i,d,p] for d in DCs) <= Supply[i,p] 
            for i in Plants for p in Products), name = 'Supply')

    # DC capacity constraints
    c2 = m.addConstrs(
        (gp.quicksum(x[i,d,p] for i in Plants for p in Products) <= DCCaps[d]
            for d in DCs), name = 'Capacity')

    # demand constraints
    c3 = m.addConstrs(
        (gp.quicksum(x[d,j,p] for d in DCs) == Demand[j,p] 
            for j in Destinations for p in Products), name = 'Demand')

    # flow balance
    c4 = m.addConstrs(
        (gp.quicksum(x[i,d,p] for i in Plants) == gp.quicksum(x[d,j,p] for j in Destinations)
            for p in Products for d in DCs), name='Flow Balance')
    
    # DC setup constraints
    c5 = m.addConstrs(
        (gp.quicksum(x[i,d,p] for i in Plants for p in Products) <= total_demand * y[d]
            for d in DCs), name='Flow Balance')
                    
    
    # Save model for inspection/debugging
    m.write('models/model_1.lp')

    # Solve the model
    m.optimize()

    # Print optimal solutions if found
    if m.status == GRB.Status.OPTIMAL:
        print("\nOptimal Solution:")
        print(f"Total Costs = {m.objVal}")
        total_cost = m.objVal
        Cost_transport = gp.quicksum(x[i,j,p]*TransCost[i,j,p]*Distances[i,j] for i,j in Vars_distances for p in Products).getValue()
        Cost_setup = gp.quicksum(y[d]*DCCosts[d] for d in DCs).getValue()
        print(f"Shipping cost = {Cost_transport}")
        print(f"Setup up cost = {Cost_setup}")
        
        print('\n')
        
        titles = ['Estabilished Distribution', 'Shpping Arragements']
        print_lines = [[] for i in range(len(titles))]
        if details:
            ti = 0
            print_lines[ti].append('=')
            tmp = []
            for d in DCs:
                if y[d].x > 0:
                    tmp.append(f"Distribution Center {d:10} is estabilished")
            if len(tmp) > 0:
                print_lines[ti].extend(tmp)
            else:
                print_lines[ti].append('No Distribution Center is estabilished')

            ti += 1
            for i, j in Vars_distances:
                tmp = []
                for p in Products:
                    if x[i,j,p].x > 0:
                        tmp.append(f"  Transport {int(x[i,j,p].x):3d} units of product {p}")
                if len(tmp) > 0:
                    print_lines[ti].append('=')
                    print_lines[ti].append(f'from {i:10s} to {j}')
                    print_lines[ti].extend(tmp)
            
            print_tables(titles, print_lines)
    
            
        sol_trans = {'Src': [], 'Dest': [], 'Part': [], 'Count': []}
        sol_dc = {'Setupped Destribution Center': [d for d in DCs if y[d].x > 0]}
        obj = {'Cost': ['Total cost', 'Total shipping cost', 'Total setup cost'], 
            'Value': [total_cost, Cost_transport, Cost_setup]}
        for i, j in Vars_distances:
            tmp = []
            for p in Products:
                if x[i,j,p].x > 0:
                    sol_trans['Src'].append(i)
                    sol_trans['Dest'].append(j)
                    sol_trans['Part'].append(p)
                    sol_trans['Count'].append(int(x[i,j,p].x))    
        
        sheets = ['objective', 'shipping solution', 'dc solution']
        sheets = [f'{sheet}_{len(Destinations)}' for sheet in sheets]
        write2Excel('results/model 1.xlsx', [obj, sol_trans, sol_dc], sheets)
    return int(m.status == GRB.Status.OPTIMAL)

def test_model_1():
        
    # Sets or Indices
    Plants = ['Hamburg','Helsinki','London']
    DCs = ['Netherland', 'HHH']
    Destinations = ['SG','Osk','Sh']
    Products = ['Gasket', 'Oil_sep', 'Ballast']

    ## Data. They can be read from external file
    supply_data = [[250, 125,  50],
                [200, 250,  50],
                [ 50, 150, 250]]

    cap_DC_data = [1000, 1500]
    cost_DC_data = [150, 200]
            
    demand_data = [[ 50, 100,  75],
                [100,  50,  75],
                [ 25,  25, 150]]

    cost_2DC_data  = [[6, 6, 6],
                    [8, 9, 9],
                    [6, 6, 4]]
    cost_2DC_data = [cost_2DC_data, [[b + 1 for b in a] for a in cost_2DC_data]]

    cost_2Dest_data  = [[6, 6, 6],
                        [8, 9, 9],
                        [6, 6, 4]]
    cost_2Dest_data = [cost_2Dest_data, [[b - 1 for b in a] for a in cost_2Dest_data]]
    
    dist_2DC_data = [[1,2], [2,1], [1,2]]
    dist_2Dest_data = [[1,2, 1], [2,1, 2]]


    ## Parameters.
    num_plants = len(Plants)
    num_dest = len(Destinations)
    num_dc = len(DCs)
    num_prod = len(Products)
    # Dictionaries to enable data to be indexed by plants and cities
    Supply = {(plant, prod) : supply_data[i][p] 
                            for i, plant in enumerate(Plants)
                            for p, prod in enumerate(Products)}

    Demand = {(dest, prod) : demand_data[j][p] 
                            for j, dest in enumerate(Destinations)
                            for p, prod in enumerate(Products)}

    TransCost = {(plant, dc, prod) : cost_2DC_data[d][i][p] 
                            for i, plant in enumerate(Plants)
                            for d, dc in enumerate(DCs)
                            for p, prod in enumerate(Products)}

    TransCost.update({(dc, dest, prod) : cost_2Dest_data[d][j][p] 
                            for d, dc in enumerate(DCs)
                            for j, dest in enumerate(Destinations)
                            for p, prod in enumerate(Products)})    
    
    DCCaps = {dc : cap_DC_data[d]
                            for d, dc in enumerate(DCs)}

    DCCosts = {dc : cost_DC_data[d]
                            for d, dc in enumerate(DCs)}
    
    Distances = {(plant, dc): dist_2DC_data[i][d]
                            for i, plant in enumerate(Plants)
                            for d, dc in enumerate(DCs)}
    Distances.update({(dc, dest): dist_2Dest_data[d][j]
                            for d, dc in enumerate(DCs)
                            for j, dest in enumerate(Destinations)})
    
    optz_model_1(Products, Plants, Supply, DCs, DCCaps, DCCosts, Destinations, Demand, TransCost, Distances)
    
if __name__ == '__main__':
    test_model_1()