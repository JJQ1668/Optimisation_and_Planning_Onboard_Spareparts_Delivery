# -*- coding: utf-8 -*-
"""
Provision of Spare Parts for Marine Vessels Maintenace at Calling Ports.
Vessels can require multiple spare parts.
Spares parts for each vessel are supplied from different plants to same 
destination.  
"""
import gurobipy as gp
from gurobipy import GRB
from print_results import *
from part_raw_data import conver_time, write2Excel

                       
def test_model_3():
    
    # Sets
    Plants = ['Hamburg','Helsinki','London']
    Dests = ['SG','Osk','Sh']
    Vessels = ['v1','v2','v3','v4','v5']
    Parts = ['sp1','sp2','sp3','sp4','sp5','sp6','sp7','sp8','sp9']
    Inspections = ['in1','in2','in3']

    # Rp[k] = list of parts required by vessel k
    Rp = { 'v1' : ['sp1','sp2','sp3','sp4','sp7'],
    'v2' : ['sp2','sp3','sp6','sp8'],
    'v3' : ['sp8','sp9'],
    'v4' : ['sp5','sp6','sp7','sp9'],
    'v5' : ['sp1','sp4','sp5'] }
    # List of (k, p) where Vessel k requires part p
    VesParts = [(k,p) for k in Vessels for p in Parts if p in Rp[k]]

    # Rn[k] = list of inspections required by vessel k
    Rn = { 'v1' : ['in1','in2'],
    'v2' : ['in2','in3'],
    'v3' : ['in2'],
    'v4' : ['in1','in3'],
    'v5' : ['in3'] }
    # List of (k,n) where Vessel k requires inspection n
    VesInspections = [(k,n) for k in Vessels for n in Inspections if n in Rn[k]]

    # Rn[k] = list of parts required by inspection n
    Rm = { 'in1' : ['sp1','sp2','sp3'],
    'in2' : ['sp4','sp5','sp6'],
    'in3' : ['sp7','sp8','sp9'] }

    # list of (k,p,n) where vessel k requires part p involved in inspection n
    VesInspectionParts = [(k,p,n) for k,n in VesInspections for p in Parts if p in Rm[n] and p in Rp[k]]

    ## Parameters
    # A[i,q] = 1 if part q is available at plant i
    data = [[1,0,1,1,0,1,0,1,1], # parts in Hamburg
            [0,1,1,0,1,1,1,0,1], # Parts in Helsinki
            [1,1,0,1,1,0,1,1,0]] # Parts in London
    A = {(i,p):data[a][b] for a,i in enumerate(Plants) for b,p in enumerate(Parts)}

    # B[j,n] = 1 if inspection n is available at dest j
    data = [[1,1,0], # inspections in SG
            [0,1,1], # inspections in Osk
            [1,1,1]] # inspections in Sh
    B = {(j,n):data[a][b] for a,j in enumerate(Dests) for b,n in enumerate(Inspections)}

    # C[i,j,p] = Cost to ship part p from plant i to destination j
    data = [[[4, 3, 3, 4, 4, 1, 5, 3, 1],   #'Hamburg','SG'
            [4, 4, 5, 4, 4, 3, 3, 2, 1],   #'Hamburg','Osk'
            [1, 5, 1, 3, 5, 5, 3, 5, 1]],  #'Hamburg','Sh'
            [[1, 1, 3, 1, 4, 4, 2, 5, 2],   #'Helsinki','SG'
            [1, 1, 5, 1, 3, 4, 2, 2, 2],   #'Helsinki','Osk'
            [5, 3, 4, 3, 1, 1, 2, 2, 5]],  #'Helsinki','Sh'
            [[5, 5, 3, 5, 5, 3, 5, 1, 2],   #'London','SG'
            [5, 1, 1, 3, 1, 3, 1, 1, 5],   #'London','Osk'
            [1, 3, 5, 1, 1, 4, 1, 5, 3]]]  #'London','Sh'
    Cship = {(i,j,p):data[a][b][c] for a,i in enumerate(Plants) for b,j in enumerate(Dests) for c,p in enumerate(Parts)}

    # Cf[i,p] = Total shipout Cost from plant i to j
    data = [[40, 10, 20, 3, 1, 2, 1, 3, 3],    #'Hamburg'
            [20, 30, 40, 2, 4, 4, 3, 3, 3],    #'Helsinki'
            [10, 40, 10, 4, 2, 4, 1, 4, 2]]    #'London'
    Cplant = {(i,j):data[a][b] for a,i in enumerate(Plants) for b,j in enumerate(Dests)}

    # Cport[j,p] = Total receiving Cost to receive part p at port j 
    data = [[2, 2, 4, 2, 2, 3, 1, 1, 4],    #'SG'
            [1, 1, 3, 1, 2, 4, 1, 1, 3],    #'Osk'
            [4, 4, 2, 2, 3, 2, 3, 4, 1]]    #'Sh'
    Cport = {(j,p):data[a][b] for a,j in enumerate(Dests) for b,p in enumerate(Parts)}

    # Cinsp[j,n] = Total Cost to conduct inspection n at port j 
    data = [[7, 10, 9],    #'SG'
            [9, 10, 9],    #'Osk'
            [7, 10, 8]]    #'Sh'
    Cinsp = {(j,n):data[a][b] for a,j in enumerate(Dests) for b,n in enumerate(Inspections)}


    # T[i,j] = time to ship an item from plant i to destination j
    T = {('Hamburg', 'SG' ): 10,
    ('Hamburg', 'Osk'): 11,
    ('Hamburg', 'Sh' ): 12,
    ('Helsinki','SG' ): 15,
    ('Helsinki','Osk'): 16,
    ('Helsinki','Sh' ): 17,
    ('London',  'SG' ): 20,
    ('London',  'Osk'): 21,
    ('London',  'Sh' ): 22 }

    # a[j,k] = time vessel k arrives at destination j
    a = {('SG', 'v1'): 10,
    ('SG', 'v2'): 11,
    ('SG', 'v3'): 12,
    ('SG', 'v4'): 13,
    ('SG', 'v5'): 14,
    ('Osk','v1'): 12,
    ('Osk','v2'): 13,
    ('Osk','v3'): 14,
    ('Osk','v4'): 15,
    ('Osk','v5'): 16,
    ('Sh', 'v1'): 20,
    ('Sh', 'v2'): 21,
    ('Sh', 'v3'): 22,
    ('Sh', 'v4'): 23,
    ('Sh', 'v5'): 24 }

    # b[j,k] = time vessel k departs destination j
    b = {('SG', 'v1'): 12,
    ('SG', 'v2'): 13,
    ('SG', 'v3'): 14,
    ('SG', 'v4'): 15,
    ('SG', 'v5'): 16,
    ('Osk','v1'): 12,
    ('Osk','v2'): 15,
    ('Osk','v3'): 16,
    ('Osk','v4'): 17,
    ('Osk','v5'): 18,
    ('Sh', 'v1'): 22,
    ('Sh', 'v2'): 23,
    ('Sh', 'v3'): 24,
    ('Sh', 'v4'): 25,
    ('Sh', 'v5'): 26 }

    # H[j,p] = Holding cost of part p at destination j
    data = [[9, 5, 5, 5, 10, 10, 7, 5, 8],  #'SG'
            [6, 10, 8, 5, 8, 10, 6, 5, 5],  #'Osk'
            [8, 9, 10, 5, 6, 6, 8, 8, 8]]   #'Sh'
    H = {(j,p):data[a][b] for a,j in enumerate(Dests) for b,p in enumerate(Parts)}
    
    Distances = {(i,j): T[i,j] for i in Plants for j in Dests}
    VesselSchedule = {(k, j): [(a[j, k], b[j, k])] for k in Vessels for j in Dests}
    optz_model_3(Parts, Inspections, Plants, Vessels, Dests, A, B, Cship, Rm, Cplant, Distances, VesselSchedule, Rp, Rn, H, Cinsp, 1)
        
Preoutputs = None 
def getResult():
    return Preoutputs
def optz_model_3(Parts, Inspections, Plants, Vessels, Destinations, AvailabilityPartPlant, 
                 AvailabilityInspPort, TransCost, PartsInInsp, TransOutCost, Distances, 
                 VesselSchedule, VesselPartDemand, VesselInspDemand, HoldingCsot, InspectionCost, 
                 VehicleSpeed, details=True):
    
    A = AvailabilityPartPlant
    B = AvailabilityInspPort
    Dests = Destinations
    Rp = VesselPartDemand
    Rn = VesselInspDemand
    Rm = PartsInInsp
    Cplant = TransOutCost
    Cinsp = InspectionCost
    T = {(i,j): int(Distances[i,j] / VehicleSpeed) for i in Plants for j in Dests}
    Cship = {(i,j,p): int(TransCost[i,j,p]*Distances[i,j]) for i in Plants for j in Dests for p in Parts}
    # Cship = TransCost
    H = HoldingCsot
    # S = {(s['Vessel'], s['Port']): [(s['ETA_time'], s['ETD_time'])] for s in VesselSchedule}
    S = VesselSchedule
    LM = 100000
    
    Vessels = list(set(Rp.keys()).union(set(Rn.keys())))
    Dests = list(set([j for k,j in S.keys() if k in Vessels]).intersection(set(Dests)))
    Parts = list(set([p for k,v in Rp.items() if k in Vessels for p in v]).intersection(set(Parts)))
    Inspections = list(set([n for k,v in Rn.items() for n in v]).intersection(set(Inspections)))
    Plants = list(set([i for i in Plants for p in Parts if A[i,p] == 1]))
    
    VesDParts = [(k,p) for k in Vessels for p in Rp[k] if p in Parts]
    PortVess = [(j,k) for k in Vessels for j in Dests if (k,j) in S.keys()]
    VesDInsps = [(k,n) for k in Vessels for n in Rn[k] if n in Inspections]
    PortVesDInsps = [(j,k,n) for j,k in PortVess for n in Rn[k] if B[j,n] == 1 and n in Inspections]
    PortVesDParts = [(j,k,p) for j,k in PortVess for p in Rp[k] if p in Parts]
    PlantPortVesDParts = [(i,j,k,p) for i in Plants for j,k,p in PortVesDParts if A[i,p] == 1]
    PlantPorts = list(set([(i,j) for i,j,k,p in PlantPortVesDParts]))
    VesSch = [(j,k,a,b,p) for j,k,p in PortVesDParts for a,b in S[k,j]]
    
    DParts_VesDInsp = {(k,n): list(set(Rm[n]).intersection(set(Rp[k]))) for k,n in VesDInsps}
    DParts_Ves = {}
    for k,p in VesDParts:
        if k in DParts_Ves:
            DParts_Ves[k].append(p)
        else:
            DParts_Ves[k] = [p]
    DInsps_Ves = {}
    for k,n in VesDInsps:
        if k in DInsps_Ves:
            DInsps_Ves[k].append(n)
        else:
            DInsps_Ves[k] = [n]
    
    Ports_Ves = {}
    for j, k in PortVess:
        if k in Ports_Ves:
            Ports_Ves[k].append(j)
        else:
            Ports_Ves[k] = [j]
    Vess_Port = {}
    for j, k in PortVess:
        if j in Vess_Port:
            Vess_Port[j].append(k)
        else:
            Vess_Port[j] = [k]
    Ports_VesIns = {}
    for k, v in Ports_Ves.items():
        for n in set(Rn[k]).intersection(Inspections):
            Ports_VesIns[k,n] = [j for j in v if B[j,n] == 1]
    VesDParts_PlantPort = {}
    for i,j,k,p in PlantPortVesDParts:
        if (i,j) in VesDParts_PlantPort:
            VesDParts_PlantPort[(i,j)].append((k,p))
        else:
            VesDParts_PlantPort[(i,j)] = [(k,p)]
    PlantPorts_VesDPart = {}
    for i,j,k,p in PlantPortVesDParts:
        if (k,p) in PlantPorts_VesDPart:
            PlantPorts_VesDPart[k,p].append((i,j))
        else:
            PlantPorts_VesDPart[k,p] = [(i,j)]
    PlantDParts_PortVes = {}
    for i,j,k,p in PlantPortVesDParts:
        if (j,k) in PlantDParts_PortVes:
            PlantDParts_PortVes[j,k].append((i,p))
        else:
            PlantDParts_PortVes[j,k] = [(i,p)]
    Plants_DPart = {p: [i for i in Plants if A[i,p] == 1] for p in Parts}
    
    DInsps_PortVes = {}
    for j, k in PortVess:
        DInsps_PortVes[j,k] = [n for n in DInsps_Ves[k] if B[j,n] == 1]
    
    
    # Ports_Ves = {k:[j for j,k1 in PortVess if k == k1] for k in set([k2 for j,k2 in PortVess])}
    
    ## Model
    m = gp.Model("Problem_3")
    # m.setParam('NonConvex', 2)

    ## Decision Variables
    # x[i,j,k,p] = 1 if part p for vessel k is shipped from plant i to dest j
    #            = 0 otherwise.
    x = m.addVars(PlantPortVesDParts, vtype=GRB.BINARY, name='x')

    # y[j,k] = 1 if vessel k is supplied at least one part at destination j
    y = m.addVars(PortVess, vtype=GRB.BINARY, name='y')

    # y2[j,k,n] = 1 if vessel k conducts inspection n at destination j
    y2 = m.addVars(PortVesDInsps, vtype=GRB.BINARY, name='y2')
    # tn[j,k,n] = performing time of inspection n by vessel k at destination j
    tn = m.addVars(PortVesDInsps, vtype=GRB.INTEGER, name='TN')

    # t[j,k,p] = Arrival time of part p for vessel k at destination j
    t = m.addVars(PortVesDParts, vtype=GRB.INTEGER, name='t')

    # w[j,k,p] = Waiting time of part p for vessel k at destination j
    w = m.addVars(PortVesDParts, vtype=GRB.INTEGER, name='w')

    # z[i,j] = 1 if there is at least one part sent from plant i to port j.
    z = m.addVars(PlantPorts, vtype=GRB.BINARY, name='z')

    # Objective function
    # Minimize the total shipping and holding costs
    m.setObjective(
    gp.quicksum(Cship[i,j,p]*x[i,j,k,p] + H[j,p]*w[j,k,p]*x[i,j,k,p]
                    for i,j,k,p in PlantPortVesDParts) + \
    gp.quicksum(Cplant[i,j]*z[i,j] for i,j in PlantPorts) + \
    gp.quicksum(Cinsp[j,n]*y2[j,k,n] for j,k,n in PortVesDInsps),
                            GRB.MINIMIZE)

    # Constraints
    # A part for a vessel is shipped only from one plant to one destination
    c1 = m.addConstrs(
            (gp.quicksum(x[i,j,k,p] for i,j in PlantPorts_VesDPart[k,p]) == 1
            for k,p in VesDParts), name='Single_supply_destination')

    # Compute y[j,k] from x[i,j,k,p]
    c2 = m.addConstrs(
            (gp.quicksum(x[i,j,k,p] for i in Plants_DPart[p]) <= y[j,k]
            for j,k,p in PortVesDParts), name='Service_location' )

    # Consolidate all parts for vessel k to one destination
    c3 = m.addConstrs(
            (gp.quicksum(y[j,k] for j in Ports_Ves[k]) == 1
            for k in Vessels), name='Consolidate')

    # Parts shipped must be available at the plants
    c4 = m.addConstrs(
            (x[i,j,k,p] <= A[i,p]
            for i,j,k,p in PlantPortVesDParts), name='Availability')

    # Arrival time for an item for vessel k at destination j
    c5 = m.addConstrs(
            (t[j,k,p] == gp.quicksum(T[i,j]*x[i,j,k,p] for i in Plants_DPart[p])
            for j,k,p in PortVesDParts), name='Arrival_time')
            
    a1 = m.addVars(VesSch, vtype=GRB.BINARY, name='a1')
    c61 = m.addConstrs(
            ( a1[j,k,a,b,p] * LM * y[j,k] >= (t[j,k,p] + w[j,k,p] - a + 1) * y[j,k]
                for j,k,a,b,p in VesSch), name='c61')
    c62 = m.addConstrs(
            ( (a1[j,k,a,b,p] - 1) * LM * y[j,k] <= (t[j,k,p] + w[j,k,p] - a) * y[j,k]
                for j,k,a,b,p in VesSch), name='c62')
    b1 = m.addVars(VesSch, vtype=GRB.BINARY, name='b1')
    c63 = m.addConstrs(
            ( b1[j,k,a,b,p] * LM * y[j,k] >= (b - t[j,k,p] + 1) * y[j,k]
                for j,k,a,b,p in VesSch), name='c63')
    c64 = m.addConstrs(
            ( (b1[j,k,a,b,p] - 1) * LM * y[j,k] <= (b - t[j,k,p]) * y[j,k] 
                for j,k,a,b,p in VesSch), name='c64')
    c66 = m.addConstrs(
            (gp.quicksum([a1[j,k,a,b,p] * b1[j,k,a,b,p] for a,b in S[k,j]]) >= 1 for j,k,p in PortVesDParts)
                ,name='c66')

    # c8 = m.addConstrs(
    # (x[i,j,k,p] <= z[i,j]
    #         for i,j,k,p in PlantPortVesDParts), name='c8')

    # Inspection conducted must be avaiable at dest j
    c9 = m.addConstrs(
            (y2[j,k,n] <= B[j,n] for j,k,n in PortVesDInsps), name='Insp_Availability')

    # Each Inspection required by vessel k must perform once
    c10 = m.addConstrs(
            (gp.quicksum(y2[j,k,n] for j in Ports_VesIns[k, n]) == 1
            for k,n in VesDInsps), name='Insp_Count')

    # Each part required by vessel k and involved by inspection n must be supplied before n conducted
    val1 = m.addVars(PortVesDParts, vtype=GRB.INTEGER, name='val1')
    # val1 supplying time of part p for vessel k at port j
    c112 = m.addConstrs((val1[j,k,p] == y[j,k] * (t[j,k,p] + w[j,k,p])
                    for j,k,p in PortVesDParts), name="c112") 

    # tn is inspection n performing time by k at port j2, it should be greater than all involved part supplying times
    c11 = m.addConstrs((tn[j,k,n] >= val1[jj,k,p] * y2[j,k,n]
                        for j,k,n in PortVesDInsps for p in DParts_VesDInsp[k,n] for jj in Ports_Ves[k]), name="c11") 
    
    # tn also should be less than the last departure time of port j
    last_departure_time = {(j,k): max([b for a,b in S[k,j]]) for j,k in PortVess}
    c12 = m.addConstrs((tn[j,k,n] <= last_departure_time[j,k] * y2[j,k,n]
                    for j,k,n in PortVesDInsps), name="c12") 
    
    #tn also should be during a calling
    pas = [(j,k,n,a,b) for j,k,n in PortVesDInsps for a,b in S[k,j]]
    pasII = {(j,k,n,a,b): str(i) for i,(j,k,n,a,b) in enumerate(pas)}
    pasIds = [str(i) for i in range(len(pas))]
    val2 = m.addVars(pasIds, vtype=GRB.BINARY, name='val2')
    val3 = m.addVars(pasIds, vtype=GRB.BINARY, name='val3')
    val4 = m.addVars(pasIds, vtype=GRB.BINARY, name='val4')
    val5 = m.addVars(pasIds, vtype=GRB.BINARY, name='val5')
    c131 = m.addConstrs((val2[str(i)] * LM >= (tn[j,k,n] - a + 1) 
                         for i,(j,k,n,a,b) in enumerate(pas)), name='c131')
    c132 = m.addConstrs(((val2[str(i)] - 1) * LM  <= (tn[j,k,n] - a)  
                         for i,(j,k,n,a,b) in enumerate(pas)), name='c132')
    c133 = m.addConstrs((val3[str(i)] * LM  >= (b - tn[j,k,n] + 1) 
                         for i,(j,k,n,a,b) in enumerate(pas)), name='c133')
    c134 = m.addConstrs(((val3[str(i)] - 1) * LM  <= (b - tn[j,k,n]) 
                         for i,(j,k,n,a,b) in enumerate(pas)), name='c134')
    c135 = m.addConstrs((val4[tr] == val2[tr] * val3[tr] for tr in pasIds), name='c135')
    c136 = m.addConstrs((val5[str(ii)] == val4[str(ii)] * y2[j,k,n] for ii,(j,k,n,a,b) in enumerate(pas)), name='c136')
    val6 = {(k,n): [val5[pasII[j,k,n,a,b]] for j in Ports_VesIns[k,n] for a,b in S[k,j] ] for k,n in VesDInsps}
    c137 = m.addConstrs((gp.quicksum(val6[k,n]) == 1 for k,n in VesDInsps), name='c137')
    
    # # inpsection time limitation
    # c137 = m.addConstrs((tn[j,k,n] <= 24 * 30 * 9 for j,k,n in PortVesDInsps), name='c138')

    # Save model for inspection/debugging
    m.write('models/model_3.rlp')
    
    m.Params.timeLimit = 120
    m.Params.MIPFocus = 3
    # m.Params.ImproveStartGap = 0.1

    # MIPFocus
    # ImproveStartTime 
    # ImproveStartGap 
    # TimeLimit
    # Method

    # Solve the model
    m.optimize()

    # Print optimal solutions if found
    if m.status == GRB.Status.OPTIMAL:
        print("\nOptimal Solution:")
        print(f"Obj Value = {m.objVal}")

        Cost_shipping = gp.quicksum(Cship[i,j,p]*x[i,j,k,p]
                                for i,j,k,p in PlantPortVesDParts).getValue()
        Cost_holding = gp.quicksum(H[j,p]*w[j,k,p]*x[i,j,k,p]
                                for i,j,k,p in PlantPortVesDParts).getValue()
        Cost_plant = gp.quicksum(Cplant[i,j]*z[i,j] 
                                   for i,j in PlantPorts).getValue()
        Cost_Inspection = gp.quicksum(Cinsp[j,n]*y2[j,k,n] 
                                      for j,k,n in PortVesDInsps).getValue()
        
        total_cost = m.objVal
        

        print(f"Total shipping cost = {Cost_shipping}")
        print(f"Total holding cost  = {Cost_holding}")
        print(f"Total shiping out cost  = {Cost_plant}")
        print(f"Total inspection cost  = {Cost_Inspection}")

        print("\n")               
        
        if details:
            titles = ['Vessel Supplying and Inspection', 'Port supplying', 'Inspections at Port']
            print_lines = [[] for i in range(len(titles))]
            
            ti = 0
            for k in Vessels:
                for j in Ports_Ves[k]:
                    if y[j,k].x == 1:
                        print_lines[ti].append('=')
                        print_lines[ti].append(f"Vessel {k}:") 
                        for a,b in S[k, j]:
                            if a <= t[j,k,DParts_Ves[k][0]].x + w[j,k,DParts_Ves[k][0]].x and b >= t[j,k,DParts_Ves[k][0]].x:
                                print_lines[ti].append(f"  Arrives {j:6s} at {a:5d}, Departs at {b}")
                                print_lines[ti].append(f"  Parts supplied {DParts_Ves[k]}")
                                for i,p in PlantDParts_PortVes[j,k]:
                                    if x[i,j,k,p].x == 1:
                                        print_lines[ti].append(f"    {p:17s} from {i:10s} arrives {j:6s} at {t[j,k,p].x:5.0f}, waiting time {w[j,k,p].x}")
                                        
                
                print_lines[ti].append('-')
                tmp = []
                for j in Ports_Ves[k]:
                    tmp1 = []
                    for n in DInsps_PortVes[j, k]:
                        if y2[j,k,n].x == 1:
                            tmp1.append(f'    {n:30s} at {tn[j,k,n].x} after supplied time {max([val1[jj,k,p].x for jj in Ports_Ves[k]])}')
                            for a,b in S[k, j]:
                                if tn[j,k,n].x >= a and tn[j,k,n].x <= b:
                                    tmp1.append(f'      When arrive at {a:5d} and departures at {b}')
                                #     tmp1.append(f'      {val2[pasII[j,k,n,a,b]].x}, {val3[pasII[j,k,n,a,b]].x}++ When arrive at {a:5d} and departures at {b}')
                                # else:
                                #     tmp1.append(f'      {val2[pasII[j,k,n,a,b]].x}, {val3[pasII[j,k,n,a,b]].x}-- When arrive at {a:5d} and departures at {b}')
                    if len(tmp1) > 0:
                        tmp.append(f'  At {j:6s}')
                        tmp.extend(tmp1)
                if len(tmp) > 0:
                    print_lines[ti].append(f'Inspections of {k}')
                    print_lines[ti].extend(tmp)

            ti += 1
            for j in Dests:
                tmp  = []
                for k in Vess_Port[j]:
                    for i,p in PlantDParts_PortVes[j,k]:
                        if x[i,j,k,p].x == 1:
                            tmp.append(f"  {p:17s} for {k} from {i:10s} arrives {j:6s} at {t[j,k,p].x}")
                if len(tmp) > 0:
                    print_lines[ti].append('=')
                    print_lines[ti].append(f"Port {j:6s}:") 
                    print_lines[ti].extend(tmp)
                        
            ti += 1
            for j in Dests:
                tmp = []
                for k in Vess_Port[j]:
                    for a,b in S[k,j]:
                        ins = []
                        for n in DInsps_PortVes[j,k]:
                            if y2[j,k,n].x == 1 and tn[j,k,n].x >= a and tn[j,k,n].x <= b:
                                ins.append(n)
                        if len(ins) > 0:
                            tmp.append(f'  {k} arrives {j:6s} at {a:5d} and departures at {b}')
                        for n in ins:
                            tmp.append(f'    performs {n:30s} at {tn[j,k,n].x}')
                if len(tmp) > 0:
                    print_lines[ti].append('=')
                    print_lines[ti].append(f"Port {j:6s}:") 
                    print_lines[ti].extend(tmp)
                              
            print_tables(titles, print_lines)
    
        sol_trans = {'Vessel': [], 'Port': [], 'Plant': [], 'Part': [], 'Arrival time': [], 'Waiting time':[], 'Supplied time': []}
        sol_insp = {'Vessel': [], 'Port': [], 'Inspection': [], 'Inspecting time': [], 'ETA': [], 'ETD': []}
        obj = {'Cost': ['Total cost', 'Total shipping cost', 'Total holding cost', 'Total ship out cost', 'Total inspection cost'], 
            'Value': [total_cost, Cost_shipping, Cost_holding, Cost_plant, Cost_Inspection]}
        for k in Vessels:
            for j in Ports_Ves[k]:
                if y[j,k].x == 1:
                    for i,p in PlantDParts_PortVes[j,k]:
                        if x[i,j,k,p].x == 1:
                            sol_trans['Vessel'].append(k)
                            sol_trans['Port'].append(j)
                            sol_trans['Plant'].append(i)
                            sol_trans['Part'].append(p)
                            atime = conver_time(t[j,k,p].x)
                            wtime = int(w[j,k,p].x)
                            sol_trans['Arrival time'].append(atime)
                            sol_trans['Waiting time'].append(wtime)
                            sol_trans['Supplied time'].append(conver_time(wtime, atime))
        for k in Vessels:
            for j in Ports_Ves[k]:
                for a,b in S[k,j]:
                    for n in DInsps_PortVes[j,k]:
                        if y2[j,k,n].x == 1 and tn[j,k,n].x >= a and tn[j,k,n].x <= b:
                            sol_insp['Vessel'].append(k)
                            sol_insp['Port'].append(j)
                            sol_insp['Inspection'].append(n)
                            sol_insp['Inspecting time'].append(conver_time(tn[j,k,n].x))
                            sol_insp['ETA'].append(conver_time(a))
                            sol_insp['ETD'].append(conver_time(b))
                            
        sheets = ['objective', 'transport solution', 'inspection solution']
        sheets = [f'{sheet}_{len(Vessels)}' for sheet in sheets]
        write2Excel('results/model 3.xlsx', [obj, sol_trans, sol_insp], sheets)
        global Preoutputs
        Preoutputs = [obj, sol_trans, sol_insp]
    return int(m.status == GRB.Status.OPTIMAL)
        
                                
if __name__ == '__main__':
    test_model_3()
                    
    