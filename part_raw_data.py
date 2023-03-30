# 1.导入pandas模块
import pandas as pd
import os, sys
from dataclasses import make_dataclass
import random
import re
from datetime import datetime, timedelta

global_names = []
ExcelWriterParams = [['w', 'error'], ['a', 'replace']]
record_columns = ['Vessel', 'Port', 'ETA_time', 'ETD_time']

min_year_of_record = 2020
start_time = datetime(year=2022, month=1, day=1)
end_time = datetime(year=2023, month=1,day=1)

def conver_time(hours, start=start_time):
    return start + timedelta(hours=hours)
             
def parse_data(filename):
    global global_names
    df = pd.read_excel(filename)
    # print(dir(df.iloc[4][0]))
    # print(df.iloc[4][0] == 'ISM SHIP MANAGEMENT PTE LTD')

    num_row = df.shape[0]
    num_col = df.shape[1]
    # for i in range(num_row):
    #     for j in range(num_col):
    #         if df.iloc[i][j] == 'CST':
    #             print(df.loc[:20])
    #             break
    #     if df.iloc[i][j] == 'CST':
    #         break
    start_indexes = None
    for i in range(num_row):
        for j in range(num_col):
            if df.iloc[i][j] == 'Week':
                start_indexes = (i,j)
                break
        if start_indexes is not None:
            break
    if start_indexes is None:
        print(filename)
        return []
    start_row, start_col = start_indexes
    port_start_col = num_col
    col_names = []
    vessel_name = None
    for j in range(start_col, num_col):
        if df.iloc[start_row + 1].isnull()[j] and not df.iloc[start_row].isnull()[j]:
            if df.iloc[start_row][j] in ['Vessel', 'Ves', 'VESSEL']:
                ii = start_row + 1
                while ii < num_row:
                    if not df.iloc[ii].isnull()[j]:
                        vessel_name = df.iloc[ii][j]
                        break
                    else:
                        ii += 1
            col_names.append(df.iloc[start_row][j])
        else:
            if 'Port' == df.iloc[start_row][j] or 'PORT' == df.iloc[start_row][j]:
                port_start_col = j + 1
            col_names.append(None)
            break
    if vessel_name == 'CST':
        record_h = 6
    else:
        record_h = 4
    num_records = int((num_row - start_row) / record_h)
    vessel_port_records = list()
    for i in range(num_records):
        record_i = start_row + i * record_h
        record_j = port_start_col
        v = df.iloc[record_i + 2][start_col]
        record_base = {}
        for j in range(record_j):
            if col_names[j] is not None:
                record_base[col_names[j]] = df.iloc[record_i + 2][j] 
            else:
                break
        row_names = ['Port', 'Terminal', 'ETA', 'ETD']
        record = record_base.copy()
        for j in range(record_j, num_col):
            if df.iloc[record_i].isnull()[j]:
                continue
            record = record_base.copy()
            if vessel_name == 'CST':
                record[row_names[0]] = df.iloc[record_i + 0][j] 
                record[row_names[1]] = df.iloc[record_i + 1][j] 
                def concatenateTs(a,b):
                    v1 = df.iloc[a][b]
                    v2 = df.iloc[a + 1][b]
                    if isinstance(v1, datetime):
                        v1 = v1.strftime('%H%M')
                    if isinstance(v2, datetime):
                        v2 = v2.strftime('%y/%m/%d')
                    if not df.iloc[a].isnull()[b]:
                        c = str(v2) + ' ' + str(v1)
                    else:
                        c = str(v2)
                    return c
                if not df.iloc[record_i + 3].isnull()[j]:
                    record[row_names[2]] = concatenateTs(record_i + 2, j)
                if not df.iloc[record_i + 3].isnull()[j]:
                    record[row_names[3]] = concatenateTs(record_i + 4, j)
            else:
                for k in range(4):
                    if not df.iloc[record_i + k].isnull()[j]:
                        record[row_names[k]] = df.iloc[record_i + k][j] 
            if len(record) > len(record_base):
                flag = True
                for k,v in record.items():
                    flag = flag and (not isinstance(v, str) or len(v) <= 20 and len(v) > 0)
                if flag:
                    vessel_port_records.append(record)
    global_names += [ n for n in col_names + row_names if n is not None and n not in global_names]
    return vessel_port_records

reg = '^' + ','.join([f'(?P<{k}>[^,]*)' for k in record_columns])
reg_time = r'ST|ND|RD|TH|LT'
fstrs = {}
# 1642LT/01ST-JAN-2022 0800LT/18ST APR-2022
fstrs['CFD'] = ['%H%M/%d-%b-%Y', '%H%M/%d %b-%Y']
# 1030LT/05/04
fstrs['CSL'] = ['%H%M/%d/%m']
# 01-JAN-2022\n1312LT 01-JAN-2022\nLT
fstrs['ISP'] = ['%d-%b-%Y\n%H%M', '%d-%b-%Y   ', '%d-%b-0%y\n%H%M', '%d-%B-%Y\n%H%M', '%d%b-%Y\n%H%M']
# 0948LT/29/04   1630LT/04/19
fstrs['CSC'] = ['%H%M/%d/%m', '%H%M/%m/%d', ['%m/%d', '%d/%m']]
# 0830LT/11/JAN 12/DEC
fstrs['CSR'] = ['%H%M/%d/%b', '%d/%b', '%H%M%d/%b']
# 04TH-JAN-2022
fstrs['CCT'] = ['%d-%b-%Y']
# 1900/26-SEP
fstrs['IVC'] = ['%H%M/%d-%b']
# 1200\n23-May-22
fstrs['CTR'] = ['%H%M\n%d-%b-%y']
# 1200\n23-May-22
fstrs['CST'] = ['%y/%m/%d %H%M', '%y/%m/%d']
# 2022122113
fstrs['Common'] = '%Y%m%d%H'
base_year = 2022
time_deviation1 = timedelta(days = 180) # half year
time_deviation2 = timedelta(seconds = 3600 * 4) # 4 hours

def parse_times(records):
    ii, jj, kk = 0, 0, 0
    rrecords = []
    for vr in records:
        vn = vr[0]['Vessel']
        pret = None
        rs = []
        for r in vr:
            ta, td = r['ETA'], r['ETD']                
            t_strs = [ta, td]
            try:
                ts = [None, None]
                for l in range(2):
                    t_strs[l] = re.sub(reg_time, '', t_strs[l])
                    t_strs[l] = re.sub(r'SEPT', 'SEP', t_strs[l])
                    if t_strs[l][0] == '/':
                         t_strs[l] =  t_strs[l][1:]
                    if vn != 'CSC':
                        offset = 1
                        if vn == 'CST':
                            pass
                        while len(fstrs[vn]) >= offset:
                            try:
                                ts[l] = datetime.strptime(t_strs[l], fstrs[vn][offset - 1])
                                break
                            except ValueError:
                                offset += 1
                        if len(fstrs[vn]) < offset:
                            raise ValueError
                    elif vn == 'CSC':
                        offset = 0
                        if int(r['Week']) == 44:
                            pass
                        try:
                            if int(r['Week']) < 10:
                                ts[l] = datetime.strptime(t_strs[l], fstrs[vn][0])
                            else:
                                ts[l] = datetime.strptime(t_strs[l], fstrs[vn][1])
                        except ValueError:
                            offset = 1
                            while len(fstrs[vn][2]) >= offset:
                                try:
                                    ts[l] = datetime.strptime(t_strs[l], fstrs[vn][2][offset - 1])
                                    break
                                except ValueError:
                                    offset += 1
                        if len(fstrs[vn][2]) < offset:
                            raise ValueError
                    else:
                        raise ValueError
                    if ts[l] is None:
                        raise ValueError
                if r['Vessel'] == 'CSC':
                    pass
                for k in range(len(ts)):
                    ts[k] = ts[k].replace( minute=0)
                    if ts[k].year <= min_year_of_record:
                        if pret is None or pret.year <= min_year_of_record:
                            ts[k] = ts[k].replace(year = base_year)
                        else:
                            ts[k] = ts[k].replace(year = pret.year)
                        if pret is not None and (pret - ts[k]) > time_deviation1:
                            ts[k] = ts[k].replace(year = pret.year + 1)
                    pret = ts[k]
                r['ETA_time'], r['ETD_time'] = ts[0], ts[1]
                jj += 1
            except ValueError:
                # vr.remove(r)  
                # print('Unparseable Record: ', r)
                r['ETA'], r['ETD'] = t_strs[0], t_strs[1]
                print('    Unparseable Record: ', r)
                rs.append(r)  
                ii += 1  
        vrr = [rr for rr in vr if rr not in rs]
        if len(vrr) > 0:
            # solve conflict
            target = vrr[:]
            target_len = len(target)
            conf = range(target_len)
            while True:
                tt = {}
                target = sorted(target, key=lambda rr: rr['ETA_time'])
                tget = [(it['ETA_time'], it['ETD_time']) for it in target]
                for i in conf:
                    ttt = [j for j in conf if target[i]['ETA_time'] <  target[j]['ETD_time'] and j < i or target[i]['ETD_time'] >  target[j]['ETA_time'] and j > i]
                    ttt = len(ttt)
                    if target[i]['ETA_time'] >= target[i]['ETD_time']:
                        ttt += 1
                    if ttt > 0:
                        tt[i] = ttt
                if len(tt.items()) == 0:
                    break
            
                k,v = max(tt.items(), key=lambda it: it[1])
                k_r = target[k]
                # if len(tt.keys()) >= 2:
                #     print(vn, len(tt.items()), k, v, k_r)
                d2 = k_r['ETD_time'] - k_r['ETA_time']
                if d2 <= timedelta():
                    d2 = time_deviation2
                delta = random.random() * time_deviation2 / 4 - time_deviation2 / 2
                if k == target_len - 1:
                    k_r['ETA_time'] = target[k - 1]['ETD_time'] + time_deviation2 / 2 + delta
                    k_r['ETD_time'] = k_r['ETA_time'] + d2 + delta
                    continue
                if k == 0:
                    k_r['ETD_time'] = target[k + 1]['ETA_time'] - time_deviation2 / 2 + delta
                    k_r['ETA_time'] = k_r['ETD_time'] - d2 + delta
                    continue
                intervals = []
                for i in conf:
                    if i == k or i == conf[-1]:
                        continue
                    if i + 1 == k:
                        intervals.append((i, i + 2, target[i]['ETD_time'], target[i + 2]['ETA_time']))
                    else:
                        intervals.append((i, i + 1, target[i]['ETD_time'], target[i + 1]['ETA_time']))
                ini, inj, lo_bd, up_bd = max(intervals, key=lambda it: it[3] - it[2])

                # print(ini, inj, target[ini]['ETA_time'], lo_bd, up_bd)
                d1 = up_bd - lo_bd
                if d1 <= timedelta():
                    print('No way to fixed confliction')
                    raise ValueError
                d2 = min(d1 / 2, d2)
                delta = random.random() * (d1 - d2) / 4 - (d1 - d2) / 2
                k_r['ETA_time'] = lo_bd + (d1 - d2) / 2 + delta
                k_r['ETD_time'] = lo_bd + (d1 + d2) / 2 + delta
            
            if target[-1]['ETD_time'].replace(year=base_year+1) > datetime.now():
                div = base_year - target[-1]['ETD_time'].year
            else:
                div = base_year + 1 - target[-1]['ETD_time'].year
            for rr in target:
                rr['ETA_time'] = rr['ETA_time'].replace(year=rr['ETA_time'].year + div)
                rr['ETD_time'] = rr['ETD_time'].replace(year=rr['ETD_time'].year + div)            

            for rr in vrr:
                rr['ETA_time'] = rr['ETA_time'].strftime(fstrs['Common'])
                rr['ETD_time'] = rr['ETD_time'].strftime(fstrs['Common'])
            tvr = [','.join([rr[k] for k in record_columns]) for rr in vrr]
            kk += len(tvr)
            tvr = set(tvr)
            kk -= len(tvr)
            regEng = re.compile(reg)
            tvr = [regEng.match(rr).groupdict() for rr in tvr]
            for tr in tvr:
                tr['ETA_time'] = datetime.strptime(tr['ETA_time'], fstrs['Common'])
                tr['ETD_time'] = datetime.strptime(tr['ETD_time'], fstrs['Common'])
            rrecords.append(tvr)
    return rrecords, ii, jj, kk
            
def gen_valid_schedules(filename, path=None, ports=None, thd1=0, thd2=0):
    filenames = list(set(os.listdir(path)))
    all_records = []
    for fn in filenames:
        rs = parse_data(path + '/' + fn)
        if len(rs) == 0:
            print(f'    No valid records in {path}/{fn}')
            continue
        trs = []
        for b in rs:
            for n in global_names:
                if n not in b:
                    b[n] = None
            if b['Port'] is None or b['ETA'] is None or b['ETD'] is None:
                continue
            elif ports is not None:
                if b['Port'] in ports:
                    b['Port'] = ports[b['Port']]
                else:
                    print('Port', b['Port'], 'is not recorded')
            trs.append(b)
        if len(trs) > 0:
            all_records.append(trs)
        else:
            print(f'    No valid records in {path}/{fn}')
                
    global_names.extend(['ETA_time', 'ETD_time'])  
    # print(sum([len(aa) for aa in all_records])) 
    t_all_records, i, j, k = parse_times(all_records)  
    all_records_len = sum([len(aa) for aa in t_all_records])
    print(f'    Invalid time records: {i}; Corrected time records: {j};', 
          f'Duplicated records: {k}; Total Converted records: {all_records_len};',
          f'Total raw records: {sum([len(aa) for aa in all_records])}') 
    all_records = [] 
    [all_records.extend(rs) for rs in t_all_records]
    
    all_vessels = set([rs[0]['Vessel'] for rs in t_all_records])
    all_vessels_len = len(all_vessels) 
    all_ports = {v: sum([v == r['Port'] for r in all_records]) for v in set([rr['Port'] for rr in all_records])}
    all_ports_len = len(all_ports)
    
    all_records = [r for r in all_records if r['ETA_time'] >= start_time and r['ETD_time'] <= end_time]
    
    if thd1 == 0: 
        thd1 = all_ports_len
    all_ports = dict(sorted(all_ports.items(), key=lambda item: item[1], reverse=True)[:thd1])

    all_vessels = {v: sum([v == r['Vessel'] and r['Port'] in all_ports for r in all_records]) for v in all_vessels}
    v_vessels = {k:v for k,v in all_vessels.items() if v > thd2}
    v_ports = {v: sum([v == r['Port'] and r['Vessel'] in all_vessels for r in all_records]) for v in all_ports}
    v_records = [{k:v for k,v in aa.items() if k in record_columns} for aa in all_records if aa['Port'] in all_ports and aa['Vessel'] in all_vessels]


    datas = [{n:[rd[n] for rd in v_records] for n in record_columns}, 
             {'Vessel': [k for k,v in v_vessels.items()], 'Counts': [v for k,v in v_vessels.items()]}, 
             {'Port': [k for k,v in v_ports.items()], 'Counts': [v for k,v in v_ports.items()]}]
    sheets = ['vessel_schedule', 'vessels', 'ports']
    write2Excel(filename, datas, sheets)
    print(  f'    Total converted records: {all_records_len}; total sampled records: {len(v_records)}\n' +
            f'    Total involved vessels: {all_vessels_len}; total sampled vessels: {len(v_vessels)}\n' +
            f'    Total involved ports: {all_ports_len}; total sampled ports: {len(v_ports)}')
    return v_records, v_vessels, v_ports
    

def load_valid_schedules(filename, path=None, ports=None,thd1=0, thd2=0, regen=False):
    if regen:
        print(f'Generate vessel schedule data from raw data path "{path}" and save in "{filename}"')
        v_records, v_vessels, v_ports = gen_valid_schedules(filename, path, ports, thd1, thd2)
    else:
        try:
            df = pd.read_excel(filename, sheet_name='vessel_schedule')
            v_records = [{v:df.iloc[i][j] for j,v in enumerate(record_columns)} for i in range(df.shape[0])]
            df = pd.read_excel(filename, sheet_name='vessels')
            v_vessels = [df.iloc[i][0] for i in range(df.shape[0])]
            df = pd.read_excel(filename, sheet_name='ports')
            v_ports = [df.iloc[i][0] for i in range(df.shape[0])]
            print(f'Load vessel schedule from "{filename}"')
        except(FileNotFoundError, ValueError) as e:
            print(f'Can not load data from "{filename}": {e}')
            print(f'  Try to generate data ......')
            return load_valid_schedules(filename, path, ports, thd1, thd2, True)
    
    formated_records = {}
    for r in v_records:
        d = (r['ETA_time'] - start_time).total_seconds() / 3600
        r['ETA_time'] = int(d) + int(int(d) < d)
        d = (r['ETD_time'] - start_time).total_seconds() / 3600
        r['ETD_time'] = int(d) + int(int(d) < d)
        if (r['Vessel'], r['Port']) in formated_records:
            formated_records[r['Vessel'], r['Port']].append((r['ETA_time'], r['ETD_time']))
        else:
            formated_records[r['Vessel'], r['Port']] = [(r['ETA_time'], r['ETD_time'])]
    return formated_records, v_vessels, v_ports


def load_locations(filename):
    df = pd.read_excel(filename, sheet_name='ports')
    all_ports = {df.iloc[i][0]: df.iloc[i][3] for i in range(df.shape[0])}
    df = pd.read_excel(filename, sheet_name='plants')
    all_plants = [df.iloc[i][0] for i in range(df.shape[0])]
    df = pd.read_excel(filename, sheet_name='distributions')
    all_dcs = [df.iloc[i][0] for i in range(df.shape[0])]
        
    return all_ports, all_plants, all_dcs

def load_valid_distances(filename, vplants=None, vdcs=None, vports=None, regen=False):
    if regen:
        print(f'Generate distance data and save in "{filename}"')
        distances = {(pl, pt): random.randrange(3000,5000) for pt in vports for pl in vplants}
        for dc in vdcs:
            for pl in vplants:
                distances[(pl, dc)] = random.randrange(250, 550)
                for pt in vports:
                    min_d = abs(distances[(pl, dc)] - distances[(pl, pt)] + 1)
                    max_d = max(distances[(pl, dc)], distances[(pl, pt)])
                    distances[(dc, pt)] = random.randrange(min_d, max_d)
        
        datas = [{'Src': [k[0] for k,v in distances.items()], 
                           'Dest': [k[1] for k,v in distances.items()], 
                           'Dist': [v for k,v in distances.items()]}]
        sheets = ['distances']
        write2Excel(filename, datas, sheets)
    else:
        try:
            df = pd.read_excel(filename, sheet_name='distances')
            print(f'Load distance data from "{filename}"')
            distances = {(df.iloc[i][0], df.iloc[i][1]): df.iloc[i][2] 
                        for i in range(df.shape[0])}
            # if vplants is not None:
            #     distances = {k:v for k,v in distances.items() if k[0] in vplants}
            # if vdcs is not None:
            #     distances = {k:v for k,v in distances.items() if k[0] in vdcs or k[1] in vdcs}
            # if vports is not None:
            #     distances = {k:v for k,v in distances.items() if k[1] in vports}
        except(FileNotFoundError, ValueError) as e:
            print(f'Can not load data from "{filename}": {e}')
            print(f'  Try to generate data ......')
            return load_valid_distances(filename, vplants, vdcs, vports, True)
    return distances

def load_trans_cost(filename, vdistances=None, vparts=None, regen=False):
    if regen:
        print(f'Generate transportation cost data and save in "{filename}"')
        costs = {(src, dest, pt): random.randrange(50,200) / 1000 for src, dest in vdistances for pt in vparts}
        
        datas = [{'Src': [k[0] for k,v in costs.items()], 
                  'Dest': [k[1] for k,v in costs.items()], 
                  'Part': [k[2] for k,v in costs.items()], 
                  'Cost': [v for k,v in costs.items()]}] 
        sheets = ['trans_costs']
        write2Excel(filename, datas, sheets)
    else:
        try:
            df = pd.read_excel(filename, sheet_name='trans_costs')
            print(f'Load transportation cost data from "{filename}"')
            costs = {(df.iloc[i][0], df.iloc[i][1], df.iloc[i][2]): df.iloc[i][3] 
                     for i in range(df.shape[0])}
            if vdistances is not None:
                costs = {k:v for k,v in costs.items() if (k[0],k[1]) in vdistances}
            if vparts is not None:
                costs = {k:v for k,v in costs.items() if k[2] in vparts}
        except(FileNotFoundError, ValueError) as e:
            print(f'Can not load data from "{filename}": {e}')
            print(f'  Try to generate data ......')
            return load_trans_cost(filename, vdistances, vparts, True)
    return costs

def load_port_demands(filename, vports=None, vparts=None, regen=False):
    if regen:
        print(f'Generate port part demand data and save in "{filename}"')
        demands = {(port, part): random.randrange(1,11) for port in vports for part in vparts}
                
        datas = [{'Port': [k[0] for k,v in demands.items()], 
                  'Part': [k[1] for k,v in demands.items()], 
                  'Count': [v for k,v in demands.items()]}] 
        sheets = ['port_demands']
        write2Excel(filename, datas, sheets)
    else:
        try:
            df = pd.read_excel(filename, sheet_name='port_demands')
            print(f'Load port part demand data from "{filename}"')
            demands = {(df.iloc[i][0], df.iloc[i][1]): df.iloc[i][2] for i in range(df.shape[0])}
        except(FileNotFoundError, ValueError) as e:
            print(f'Can not load data from "{filename}": {e}')
            print(f'  Try to generate data ......')
            return load_port_demands(filename, vports, vparts, True)
    return demands

def load_vessel_demands(filename, vessels=None, vparts=None, 
                        vplants=None, vsupply=None, vports=None, 
                        vtimes=None, vschedules=None, regen=False):
    if regen:
        print(f'Generate vessel part demand data and save in "{filename}"')
        demands = {}
        for v in vessels:
            routine = [(po, etd) for (ve, po),ts in vschedules.items() if ve == v and po in vports for eta, etd in ts ]
            parts = set()
            for r in routine:
                port = r[0]
                etd = r[1]
                for part in vparts:
                    if max([int(vtimes[plant, port] <= vsupply[plant, part] * etd) for plant in vplants]) == 1:
                        parts.add(part)
            l = random.randrange(int(len(parts) / 2), len(parts) + 1)
            demands[v] = random.sample(list(parts), l)
        # demands = {v:random.sample(vparts, k=random.randrange(int(len(vparts)/3), len(vparts))) for v in vessels}
        
        datas = [{'Vessel': [k for k,v in demands.items() for vi in v], 
                  'Part': [vi for k,v in demands.items() for vi in v]}] 
        sheets = ['vessel_demands']
        write2Excel(filename, datas, sheets)
    else:
        try:
            df = pd.read_excel(filename, sheet_name='vessel_demands')
            print(f'Load vessel part demand data from "{filename}"')
            demands = {v: [df.iloc[i][1] for i in range(df.shape[0]) if df.iloc[i][0] == v and df.iloc[i][1]] for v in set([df.iloc[j][0] for j in range(df.shape[0])])}
            if vparts is not None:
                demands = {k:[i for i in v if i in vparts] for k,v in demands.items()}
            if vessels is not None:
                demands = {k:v for k,v in demands.items() if k in vessels}
        except(FileNotFoundError, ValueError) as e:
            print(f'Can not load data from "{filename}": {e}')
            print(f'  Try to generate data ......')
            return load_vessel_demands(filename, vessels, vparts, True)
    return demands

def load_vessel_insp_demands(filename, vessels=None, vdemand=None, 
                             vplants=None, vsupply=None, vinsps=None, 
                             vtimes=None, vschedule=None, insp_avail=None, 
                             vinsps_parts=None, regen=False):
    if regen:
        print(f'Generate vessel inspection demand data and save in "{filename}"')
        demands = {}
        for v in vessels:
            vports = {pt: (min([eta for eta, etd in ts]), max([etd for eta, etd in ts])) for (ve, pt), ts in vschedule.items() if ve == v}
            insps = []
            for port in vports:
                for insp in vinsps:
                    if insp in insps:
                        continue
                    if insp_avail[port, insp] == 0:
                        continue
                    insp_parts = set(vinsps_parts[insp]).intersection(set(vdemand[v]))
                    pst = []
                    for part in insp_parts:
                        st = min([max(vtimes[plant, sport], vports[sport][0]) for plant in vplants for sport in vports if vsupply[plant, part] == 1])
                        pst.append(st)
                    if len(pst) == 0:
                        continue
                    if max(pst) > vports[port][1]:
                        continue
                    # print('===========', v, port, insp, max(pst), vports[port][1], '============')
                    insps.append(insp)
            if len(insps) == 0:
                print(f'  {v} has no feasible inspection demand')
            l = random.randrange(max(int(len(insps) / 2), 1), len(insps) + 1)
            demands[v] = random.sample(insps, l)
                
            # ports_insp = {n:[j for j,nn in insp_avail if nn == n and j in vports] for n in vinsps}
            # insps = [insp for insp in vinsps if len(ports_insp[insp]) > 0]
            # random.shuffle(insps)
            # demands[v] = insps[:-1]
        
        datas = [{'Vessel': [k for k,v in demands.items() for vi in v], 
                  'Part': [vi for k,v in demands.items() for vi in v]}] 
        sheets = ['vessel_insp_demands']
        write2Excel(filename, datas, sheets)
    else:
        try:
            df = pd.read_excel(filename, sheet_name='vessel_insp_demands')
            print(f'Load vessel inspection demand data from "{filename}"')
            demands = {v: [df.iloc[i][1] for i in range(df.shape[0]) if df.iloc[i][0] == v] for v in set([df.iloc[j][0] for j in range(df.shape[0])])}
        except(FileNotFoundError, ValueError) as e:
            print(f'Can not load data from "{filename}": {e}')
            print(f'  Try to generate data ......')
            return load_vessel_insp_demands(filename, vessels, vinsps, vschedule, insp_avail, True)
    return demands

def load_holding_costs(filename, vports=None, vparts=None, regen=False):
    if regen:
        print(f'Generate holding cost data and save in "{filename}"')
        costs = {(port, part): random.randrange(1,100) / 500 + 0.1 for port in vports for part in vparts}
        
        datas = [{'Port': [k[0] for k,v in costs.items()], 
                  'Part': [k[1] for k,v in costs.items()], 
                  'Cost': [v for k,v in costs.items()]}] 
        sheets = ['holding_cost']
        write2Excel(filename, datas, sheets)
    else:
        try:
            df = pd.read_excel(filename, sheet_name='holding_cost')
            print(f'Load holding cost data from "{filename}"')
            costs = {(df.iloc[i][0], df.iloc[i][1]): df.iloc[i][2] for i in range(df.shape[0])}
        except(FileNotFoundError, ValueError) as e:
            print(f'Can not load data from "{filename}": {e}')
            print(f'  Try to generate data ......')
            return load_holding_costs(filename, vports, vparts, True)
    return costs

def load_part_at_port_availibility(filename, vports=None, vparts=None, regen=False):
    if regen:
        print(f'Generate part vs port availability data and save in "{filename}"')
        avails = {(port, part): int(random.random() < 0.7) for port in vports for part in vparts}
        
        datas = [{'Port': [k[0] for k,v in avails.items()], 
                  'Part': [k[1] for k,v in avails.items()], 
                  'Availibility': [v for k,v in avails.items()]}] 
        sheets = ['avails_part_port']
        write2Excel(filename, datas, sheets)
    else:
        try:
            df = pd.read_excel(filename, sheet_name='avails_part_port')
            print(f'Load part vs port availability data from "{filename}"')
            avails = {(df.iloc[i][0], df.iloc[i][1]): df.iloc[i][2] for i in range(df.shape[0])}
        except(FileNotFoundError, ValueError) as e:
            print(f'Can not load data from "{filename}": {e}')
            print(f'  Try to generate data ......')
            return load_part_at_port_availibility(filename, vports, vparts, True)
    return avails

def load_part_at_plant_availibility(filename, vplants=None, vparts=None, regen=False):
    if regen:
        print(f'Generate part availability data and save in "{filename}"')
        avails = {}
        for part in vparts:
            v = random.randrange(1, len(vplants) + 1)
            pls = random.sample(vplants, v)
            for pl in vplants:
                if pl in pls:
                    avails[pl, part] = 1
                else:
                    avails[pl, part] = 0

        datas = [{'Plant': [k[0] for k,v in avails.items()], 
                  'Part': [k[1] for k,v in avails.items()], 
                  'Availibility': [v for k,v in avails.items()]}] 
        sheets = ['avails_part_plant']
        write2Excel(filename, datas, sheets)
    else:
        try:
            df = pd.read_excel(filename, sheet_name='avails_part_plant')
            print(f'Load part availability data from "{filename}"')
            avails = {(df.iloc[i][0], df.iloc[i][1]): df.iloc[i][2] for i in range(df.shape[0])}
        except(FileNotFoundError, ValueError) as e:
            print(f'Can not load data from "{filename}": {e}')
            print(f'  Try to generate data ......')
            return load_part_at_plant_availibility(filename, vplants, vparts, True)
    return avails

def load_insp_at_port_availability(filename, vports=None, vinsps=None, regen=False):
    if regen:
        print(f'Generate inspection availability data and save in "{filename}"')
        avails = {}
        for insp in vinsps:
            v = random.randrange(max(int(len(vports)/2), 1), len(vports) + 1)
            pts = random.sample(vports, v)
            for pt in vports:
                if pt in pts:
                    avails[pt, insp] = 1
                else:
                    avails[pt, insp] = 0
                    
        datas = [{'Port': [k[0] for k,v in avails.items()], 
                  'Inspection': [k[1] for k,v in avails.items()], 
                  'Availibility': [v for k,v in avails.items()]}] 
        sheets = ['avails_insp_port']
        write2Excel(filename, datas, sheets)
    else:
        try:
            df = pd.read_excel(filename, sheet_name='avails_insp_port')
            print(f'Load inspection availability data from "{filename}"')
            avails = {(df.iloc[i][0], df.iloc[i][1]): df.iloc[i][2] for i in range(df.shape[0])}
        except(FileNotFoundError, ValueError) as e:
            print(f'Can not load data from "{filename}": {e}')
            print(f'  Try to generate data ......')
            return load_insp_at_port_availability(filename, vports, vinsps, True)
    return avails

def load_inspection_cost(filename, vports=None, vinsps=None, regen=False):
    if regen:
        print(f'Generate inspection cost data and save in "{filename}"')
        insp_base_cost = {insp: random.choice(range(2000,20001,100)) for insp in vinsps} 
        avails = {(port, insp): random.choice(range(0, 2001, 100)) + insp_base_cost[insp] for port in vports for insp in vinsps}

        datas = [{'Port': [k[0] for k,v in avails.items()], 
                  'Inspection': [k[1] for k,v in avails.items()], 
                  'Price': [v for k,v in avails.items()]}] 
        sheets = ['inspection_cost']
        write2Excel(filename, datas, sheets)
    else:
        try:
            df = pd.read_excel(filename, sheet_name='inspection_cost')
            print(f'Load inspection cost data from "{filename}"')
            avails = {(df.iloc[i][0], df.iloc[i][1]): df.iloc[i][2] for i in range(df.shape[0])}
        except(FileNotFoundError, ValueError) as e:
            print(f'Can not load data from "{filename}": {e}')
            print(f'  Try to generate data ......')
            return load_inspection_cost(filename, vports, vinsps, True)
    return avails

def load_dc_capacity_and_setup_cost(filename, vdcs=None, vport_demands=None, regen=False):
    if regen:
        print(f'Generate distribution data and save in "{filename}"')
        total_demand = sum([v for k,v in vport_demands.items()])
        total_cap = int(random.randrange(total_demand, 2*total_demand)/10) * 10 + 10
        ps = sorted(random.sample(range(0, total_cap, 10), k = len(vdcs) - 1) + [total_cap])
        dc_caps = {}
        off = 0
        for i,dc in enumerate(vdcs):
            dc_caps[dc] = ps[i] - off
            off = ps[i]
        dc_setup_costs = {dc: random.randrange(20000, 30001, 1000) for dc in vdcs}
        
        datas = [{'Distribution': [k for k,v in dc_caps.items()], 
                  'Capacity': [v for k,v in dc_caps.items()]}, 
                 {'Distribution': [k for k,v in dc_setup_costs.items()], 
                  'Setup Cost': [v for k,v in dc_setup_costs.items()]}] 
        sheets = ['dc_capacities', 'dc_setup_costs']
        write2Excel(filename, datas, sheets)
    else:
        try:
            df_caps = pd.read_excel(filename, sheet_name='dc_capacities')
            print(f'Load distribution data from "{filename}"')
            df_costs = pd.read_excel(filename, sheet_name='dc_setup_costs')
            dc_caps = {df_caps.iloc[i][0]:df_caps.iloc[i][1] for i in range(df_caps.shape[0]) if df_caps.iloc[i][0] in vdcs or vdcs is None}
            dc_setup_costs = {df_costs.iloc[i][0]:df_costs.iloc[i][1] for i in range(df_costs.shape[0]) if df_costs.iloc[i][0] in vdcs or vdcs is None}
        except(FileNotFoundError, ValueError) as e:
            print(f'Can not load data from "{filename}": {e}')
            print(f'  Try to generate data ......')
            return load_dc_capacity_and_setup_cost(filename, vdcs, vport_demands, True)
    return dc_caps, dc_setup_costs

def load_plant_capacity(filename, vplants=None, vparts=None, vport_demands=None, vsupply=None, regen=False):
    if regen:
        print(f'Generate plant capacity data and save in "{filename}"')
        pl_caps = {}
        for pt in vparts:
            total_demand = sum([v for (k1,k2),v in vport_demands.items() if k2 == pt])
            total_cap = int(random.randrange(total_demand, 2*total_demand)/10)*10 + 10
            pls = [k1 for (k1,k2),v in vsupply.items() if k2 == pt and k1 in vplants and v == 1]
            ps = sorted(random.sample(range(0, total_cap, 10), k = len(pls) - 1) + [total_cap])
            off = 0
            for i,pl in enumerate(pls):
                pl_caps[pl, pt] = ps[i] - off
                off = ps[i]
            for pl in set(vplants).difference(set(pls)):
                pl_caps[pl, pt] = 0
        
        datas = [{'Plant': [k[0] for k,v in pl_caps.items()], 
                  'Part':[k[1] for k,v in pl_caps.items()] , 
                  'Capacity': [v for k,v in pl_caps.items()]}] 
        sheets = ['plant_capacities']
        write2Excel(filename, datas, sheets)
    else:
        try:
            df_caps = pd.read_excel(filename, sheet_name='plant_capacities')
            print(f'Load plant capacity data from "{filename}"')
            pl_caps = {(df_caps.iloc[i][0],df_caps.iloc[i][1]):df_caps.iloc[i][2] 
                    for i in range(df_caps.shape[0]) 
                    if df_caps.iloc[i][1] in vparts or vparts is None}
        except(FileNotFoundError, ValueError) as e:
            print(f'Can not load data from "{filename}": {e}')
            print(f'  Try to generate data ......')
            return load_plant_capacity(filename, vplants, vparts, vport_demands, True)
    return pl_caps
    
    
def load_plant_shipout_cost(filename, vlines=None, regen=False):
    if regen:
        print(f'Generate plant shipout data and save in "{filename}"')
        pl_shipout = {(src, dest): random.randrange(400, 600) for src, dest in vlines}
        
        datas = [{'Src': [k[0] for k,v in pl_shipout.items()], 
                  'Dest':[k[1] for k,v in pl_shipout.items()] , 
                  'ShipOutCost': [v for k,v in pl_shipout.items()]}] 
        sheets = ['plant_shipout_costs']
        write2Excel(filename, datas, sheets)
    else:
        try:
            df_caps = pd.read_excel(filename, sheet_name='plant_shipout_costs')
            print(f'Load plant shipout data from "{filename}"')
            pl_shipout = {(df_caps.iloc[i][0],df_caps.iloc[i][1]):df_caps.iloc[i][2] 
                    for i in range(df_caps.shape[0]) 
                    if (df_caps.iloc[i][0],df_caps.iloc[i][1]) in vlines or vlines is None}
        except(FileNotFoundError, ValueError) as e:
            print(f'Can not load data from "{filename}": {e}')
            print(f'  Try to generate data ......')
            return load_plant_shipout_cost(filename, vlines, True)
    return pl_shipout

def load_carbon_emmissions(filename, vdistances=None, vparts=None, regen=False):
    if regen:
        print(f'Generate carbon emission data and save in "{filename}"')
        carbonEm = {(src, dest, pt): random.randrange(50,200) / 100 for src, dest in vdistances for pt in vparts}
        
        datas = [{  'Src': [k[0] for k,v in carbonEm.items()], 
                    'Dest': [k[1] for k,v in carbonEm.items()], 
                    'Part': [k[2] for k,v in carbonEm.items()], 
                    'Carbon Emission': [v for k,v in carbonEm.items()]}] 
        sheets = ['carbon_emissions']
        write2Excel(filename, datas, sheets)
    else:
        try:
            df = pd.read_excel(filename, sheet_name='carbon_emissions')
            print(f'Load carbon emission data from "{filename}"')
            carbonEm = {(df.iloc[i][0], df.iloc[i][1], df.iloc[i][2]): df.iloc[i][3] 
                        for i in range(df.shape[0])}
            if vdistances is not None:
                carbonEm = {k:v for k,v in carbonEm.items() if (k[0],k[1]) in vdistances}
            if vparts is not None:
                carbonEm = {k:v for k,v in carbonEm.items() if k[2] in vparts}
        except(FileNotFoundError, ValueError) as e:
            print(f'Can not load data from "{filename}": {e}')
            print(f'  Try to generate data ......')
            return load_carbon_emmissions(filename, vdistances, vparts, True)
    return carbonEm
    

def load_parts_inspections(filename, pl=0):
    df = pd.read_excel(filename, 'Parts')
    num_row = df.shape[0]
    parts = [df.iloc[i][0] for i in range(num_row)]
    # types = [df.iloc[i][1] for i in range(num_row)]
    # ix = list(range(num_row))
    # random.shuffle(ix)
    # write2Excel(filename, [{'Part': [parts[i] for i in range(num_row)], 
    #                         'type': [types[i] for i in range(num_row)]}], ['Parts'])
    
    
    df = pd.read_excel(filename, 'Inspection')    
    num_row = df.shape[0]
    inspections = [df.iloc[i][0] for i in range(num_row)]
    df = pd.read_excel(filename, 'Inspected_parts')
    num_row = df.shape[0]
    ks = set([df.iloc[i][0] for  i in range(num_row)])
    insp_pts = {k:[df.iloc[i][1]for i in range(num_row) if df.iloc[i][0] == k] for k in ks}
    
    if pl == 0:
        pl = len(parts)
    parts = parts[:pl]
    insp_pts = {k:[df.iloc[i][1]for i in range(num_row) if df.iloc[i][0] == k and df.iloc[i][1] in parts] for k in ks if k in inspections}
    inspections = [i for i in inspections if i in insp_pts]
        
    b = {p:[0] * len(inspections) for p in parts}
    for k,v in insp_pts.items():
        for p in v:
            b[p][inspections.index(k)] = 1
    for k,v in b.items():
        v.append(sum(v))
    bin_dict = {"Inspections": inspections[:]}
    bin_dict["Inspections"].append('Sum')
    bin_dict.update(b)
    write2Excel(filename, [bin_dict], ['insp_part_bin'])
        
    return parts, inspections, insp_pts

def setRegenDefault(flag):
    funcs = [v for k,v in globals().items() if k[:5] == 'load_' and 'regen' in v.__code__.co_varnames]
    for func in funcs:
        dfs = list(func.__defaults__)
        i = list(func.__code__.co_varnames).index('regen') - func.__code__.co_argcount + len(func.__defaults__)
        dfs[i] = flag
        func.__defaults__ = tuple(dfs)
        # print(func.__code__.co_argcount, func.__defaults__, len(func.__defaults__))
        
    
def write2Excel(path, datas, sheets):
    if os.path.exists(path):
        with  pd.ExcelWriter(path, mode='a', if_sheet_exists='replace') as writer:
            for i in range(len(datas)):
                df = pd.DataFrame(datas[i])
                df.to_excel(writer, sheets[i], index=False)
    else:
        with  pd.ExcelWriter(path, mode='w') as writer:
            for i in range(len(datas)):
                df = pd.DataFrame(datas[i])
                df.to_excel(writer, sheets[i], index=False)

def test_data():
    
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
        
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        
        all_ports, all_plants, all_dcs = load_locations(locations)
        v_records, v_vessels, v_ports = load_valid_schedules(schedules, path=raw_data_path, ports=all_ports, thd1=10)
        all_distances = load_valid_distances(locations, vplants=all_plants, vdcs=all_dcs, vports=v_ports)
        all_parts, all_insps, all_insp_pts = load_parts_inspections(parts, pl=15)
        trans_costs = load_trans_cost(costs, vdistances=all_distances, vparts=all_parts)
        vessel_demands = load_vessel_demands(demands, vessels=v_vessels, vparts=all_parts)
        port_demands = load_port_demands(demands, vports=v_ports, vparts=all_parts)
        holding_costs = load_holding_costs(costs, vports=v_ports, vparts=all_parts)
        part_plant_avails = load_part_at_plant_availibility(availabilities, vplant=all_plants, vparts=all_parts)
        part_plant_shipout = load_plant_shipout_cost(availabilities, vplants=all_plants, vparts=all_parts)
        insp_port_avails = load_insp_at_port_availability(availabilities, vports=v_ports, vinsps=all_insps)
        insp_costs = load_inspection_cost(costs, vports=v_ports, vinsps=all_insps)
        vessel_insp_demands = load_vessel_insp_demands(demands, vessels=v_vessels, vinsps=all_insps)
        dc_caps, dc_costs = load_dc_capacity_and_setup_cost(capacities, vdcs=all_dcs, vport_demands=port_demands)
        pl_caps = load_plant_capacity(capacities, vplants=all_plants, vparts=all_parts, vport_demands=port_demands)
        carbon_emissions = load_carbon_emmissions(carbon, vdistances=all_distances, vparts=all_parts)
        

if __name__ == '__main__':
    test_data()
    
    
