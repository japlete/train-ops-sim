import numpy as np
import pandas as pd
from datetime import time
from numba.typed import List

from .sim_objs import idx_tp, small_counter, pos_tp
from .event_list import time_tp

stations_sheet_name = 'Stations'
closed_circuits_sheet_name = 'Closed circuits'
open_circuits_sheet_name = 'Open circuits'
accepted_positive_marks = ['yes','y','true','t','1','1.0']

def preproc_sim_params(filename : str):
    '''
    Read input tables from 'filename' spreadsheet, return a list with the arguments for constructing a Simulation.
    The tables with the circuits, stations and itineraries are also returned for plotting, analysis and debugging.
    '''
    # Read list of stations
    stations = pd.read_excel(filename, sheet_name= stations_sheet_name, usecols= np.arange(6)).fillna(0.0)
    assert stations.iloc[:,3].min() >= 1, 'Trains capacity per station must be at least 1'
    assert stations.iloc[:,1:3].min().min() >= 0.0, 'Waiting time must be a non-negative number'
    assert not stations.iloc[:,0].duplicated().sum(), 'Duplicate station names found'
    stations.columns = ['st_name','waiter_switch_tm','passer_switch_tm','st_cap','graph_section','graph_loc']
    stations = stations.reset_index(drop= False).rename(columns= {'index' : 'st_id'}).set_index('st_name')
    
    # Process station locations and graph sections
    section_names = stations.graph_section.str.split(';', expand= True).map(lambda s : s.strip() if s is not None else None).dropna(how= 'all', axis= 1)
    section_names.columns = ['graph_sec_'+str(c) for c in section_names.columns]
    section_locs = (
        stations.graph_loc.str.split(';', expand= True)
        .map(lambda s : s.strip() if s is not None else None)
        .map(lambda s : None if s == '' else s)
        .dropna(how= 'all', axis= 1).astype(float)
    )
    assert len(section_names.columns) == len(section_locs.columns), 'Number of track sections and Km locations must match'
    section_locs.columns = ['graph_sec_'+str(c)+'_loc' for c in section_locs.columns]
    sections = pd.concat([section_names, section_locs], axis= 1).sort_index(axis= 1)
    n_sections = len(sections.columns)//2
    graph_locs = pd.DataFrame()
    for k in range(n_sections):
        aux = (
            sections[[f'graph_sec_{k}',f'graph_sec_{k}_loc']]
            .set_index(f'graph_sec_{k}', append= True)
            .unstack(-1)[f'graph_sec_{k}_loc']
        )
        if graph_locs.empty:
            graph_locs = aux.copy()
        else:
            graph_locs = graph_locs.fillna(aux)
    graph_locs = graph_locs.dropna(how= 'all', axis= 1).dropna(how= 'all', axis= 0)
    stations = stations.join(graph_locs).drop(columns= ['graph_section','graph_loc'])

    # Read closed circuits
    closed_circuits = pd.read_excel(filename, sheet_name= closed_circuits_sheet_name, usecols= np.arange(2))
    if not closed_circuits.empty:
        assert closed_circuits.iloc[:,1].min() >= 1, 'Number of trains must be at least 1'
        assert not closed_circuits.iloc[:,0].duplicated().sum(), 'Duplicate circuit names found'
    closed_circuits.columns = ['circuit_name','n_max_trains']
    closed_circuits['open'] = False

    # Read open circuits
    open_circuits = pd.read_excel(filename, sheet_name= open_circuits_sheet_name, usecols= np.arange(4))
    if not open_circuits.empty:
        assert open_circuits.iloc[:,3].min() >= 1, 'Number of trains must be at least 1'
        assert open_circuits.iloc[:,1].min() > 0.0, 'Frequency must be a positive number'
        assert open_circuits.dtypes.iloc[2] == time, 'First train time not valid time'
        assert not open_circuits.iloc[:,0].duplicated().sum(), 'Duplicate circuit names found'
    open_circuits.columns = ['circuit_name','frequency','first_tr_departure_tm','n_max_trains']
    open_circuits['open'] = True

    # Concatenate the open and closed circuit tables
    circuits = pd.concat([closed_circuits, open_circuits]).sort_values('n_max_trains', ascending= False)
    assert not circuits.empty, 'No circuits found in file'
    circuits = circuits.reset_index(drop= True).reset_index(drop= False).rename(columns= {'index' : 'circuit_id'})

    # Read each circuit itinerary, concatenate all of them into c_full
    c_full = pd.DataFrame()
    origin_st_ids = []
    cycle_tms = []
    first_dp_tms = []
    for idx,row in circuits.iterrows():
        cid = row.circuit_id
        cn = row.circuit_name
        circuit = pd.read_excel(filename, sheet_name= cn, usecols= np.arange(7)).fillna(0)
        assert circuit.iloc[:,2].min() >= 1e-5, 'Travel time must be a positive value'
        assert circuit.iloc[:,3].min() >= 0.0, 'Process time must be a non-negative value'
        assert circuit.iloc[:,4].min() >= 0.0, 'Cooldown time must be a non-negative value'
        assert circuit.iloc[:,0].isin(stations.index).all(), 'Origin station names must be contained in the Stations sheet'
        assert circuit.iloc[:,1].isin(stations.index).all(), 'Destionation station names must be contained in the Stations sheet'
        assert (circuit.iloc[:,0] != circuit.iloc[:,1]).all(), 'Origin and destionation station names must be different'
        assert (circuit.iloc[1:,0] == circuit.iloc[:,1].shift(1).iloc[1:]).all(), 'Origin station name must match the preceding destionation name'
        if not row.open:
            assert circuit.iloc[0,0] == circuit.iloc[-1,1], 'Last destionation must match first origin for closed circuits'
        circuit.columns = ['origin_st_name','dest_st_name','travel_tm','process_tm','cooldown_tm','priority_operator','double_track']
        circuit = circuit.reset_index(drop= False).rename(columns= {'index' : 'circuit_segm_idx'})
        circuit['priority_operator'] = circuit.priority_operator.map(str).str.lower().isin(accepted_positive_marks)
        circuit['double_track'] = circuit.double_track.map(str).str.lower().isin(accepted_positive_marks)
        circuit['acum_exit_tm'] = circuit.travel_tm.cumsum() + circuit.process_tm.cumsum()
        circuit['acum_entry_tm'] = circuit.acum_exit_tm - circuit.process_tm
        origin_st_ids.append(stations.st_id.loc[circuit.origin_st_name.iloc[0]])
        cycle_tms.append(circuit.acum_exit_tm.iloc[-1])
        if pd.isna(row.first_tr_departure_tm):
            first_dp_tms.append(np.nan)
        else:
            first_dp_tms.append(row.first_tr_departure_tm.hour * 60.0 + row.first_tr_departure_tm.minute + row.first_tr_departure_tm.second / 60.0)
        circuit['circuit_id'] = cid
        c_full = pd.concat([c_full, circuit.drop(columns= ['travel_tm','process_tm'])])

    # Complete the circuits df with the itinerary info
    circuits['origin_st_id'] = origin_st_ids
    circuits['cycle_tm'] = cycle_tms
    circuits['first_tr_departure_tm'] = first_dp_tms
    circuits['frequency'] *= 60.0
    circuits['n_start_trains'] = circuits.n_max_trains.astype(float).where(~circuits.open, circuits.cycle_tm / circuits.frequency).map(np.ceil).astype(int)
    circuits['frequency'] = circuits['frequency'].fillna(circuits.cycle_tm / circuits.n_start_trains)
    circuits['next_spawn'] = -1
    circuits['next_spawn'] = circuits.next_spawn.where(~circuits.open, circuits.n_start_trains*(circuits.n_start_trains < circuits.n_max_trains))

    # Join the itinerary with the station IDs to create the segment codes
    c_full = (
        c_full.join(stations.drop(columns= 'st_cap'), on= 'dest_st_name')
        .rename(columns= {'st_id' : 'dest_st_id'})
        .join(stations.st_id, on= 'origin_st_name').rename(columns= {'st_id' : 'orig_st_id'})
    )
    c_full['segm_code'] = c_full[['orig_st_id','dest_st_id']].min(axis= 1).astype(str) + '_' + c_full[['orig_st_id','dest_st_id']].max(axis= 1).astype(str)
    c_full['segm_code_dir'] = c_full['orig_st_id'].astype(str) + '_' + c_full['dest_st_id'].astype(str)
    c_full['segm_code_uni'] = np.where(c_full.double_track, c_full.segm_code_dir, c_full.segm_code)
    c_full['segm_id'] = pd.factorize(c_full.segm_code_uni, sort= True)[0]

    # Build input columns for the Segment class
    segm_circuits = c_full[['segm_id','circuit_id','circuit_segm_idx']].copy()
    segm_circuits['prev_csidx'] = np.where(segm_circuits.circuit_segm_idx > 0, 
                                 segm_circuits.circuit_segm_idx - 1, 
                                 segm_circuits.groupby('circuit_id').circuit_segm_idx.transform('max'))
    segm_circuits = (
        segm_circuits.map(idx_tp.cast_python_value)
        .groupby(['segm_id','circuit_id']).prev_csidx.apply(np.array)
        .reset_index('circuit_id').apply(tuple, axis= 1)
        .groupby(['segm_id']).apply(list).sort_index()
    )
    priority_circuits = c_full[['circuit_id','segm_id','priority_operator']].groupby(['circuit_id','segm_id']).min().iloc[:,0].unstack(0, fill_value= False).apply(np.array, axis= 1).sort_index()

    # Build input arrays for the Circuit and Station classes
    station_origins = c_full[['orig_st_id','dest_st_id']].drop_duplicates().map(idx_tp).groupby('dest_st_id').orig_st_id.apply(np.array)
    station_origins = station_origins.to_frame().join(stations.set_index('st_id').st_cap, how= 'right').sort_index()
    circuits.set_index('circuit_id', inplace= True)
    c_full.set_index('circuit_id', inplace= True)

    args_stations = List()
    args_segments = List()
    args_circuits = List()
    for st_id, row in station_origins.iterrows():
        if type(row.orig_st_id) == float:
            args_stations.append( (np.delete(np.array([idx_tp(0)]), 0), small_counter(0.0)) )
        else:
            args_stations.append( (row.orig_st_id, small_counter(row.st_cap)) )
    for k in range(len(segm_circuits)):
        aux_list = List()
        for t in segm_circuits.loc[k]:
            aux_list.append((idx_tp(t[0]), t[1]))
        args_segments.append(
            (aux_list, 
            priority_circuits.loc[k])
        )
    for k in range(len(circuits)):
        c_row = circuits.loc[k]
        origin_st_id = c_row.origin_st_id
        next_spawn = c_row.next_spawn
        frequency = c_row.frequency
        cfull_row = c_full.loc[k].sort_values('circuit_segm_idx')
        acum_exit_tm = cfull_row.acum_exit_tm.map(time_tp).to_numpy()
        cooldown_tm = cfull_row.cooldown_tm.map(time_tp).to_numpy()
        if next_spawn == -1:
            acum_exit_tm = np.concatenate([acum_exit_tm, acum_exit_tm + acum_exit_tm[-1]])
        acum_entry_tm = cfull_row.acum_entry_tm.map(time_tp).to_numpy()
        segment_ids = cfull_row.segm_id.map(idx_tp).to_numpy()
        dest_st_id = cfull_row.dest_st_id.map(idx_tp).to_numpy()
        waiter_switch_tm = cfull_row.waiter_switch_tm.map(time_tp).to_numpy()
        passer_switch_tm = cfull_row.passer_switch_tm.map(time_tp).to_numpy()
        args_circuits.append(
            (acum_exit_tm, acum_entry_tm, segment_ids, dest_st_id, waiter_switch_tm, passer_switch_tm, cooldown_tm, idx_tp(origin_st_id), idx_tp(next_spawn), time_tp(frequency))
        )

    # Build scalar arguments for the Simulation constructor
    circuits.sort_index(inplace= True)
    n_start_trains = circuits.n_start_trains.map(idx_tp).to_numpy()
    n_max_trains = circuits.n_max_trains.map(idx_tp).to_numpy()
    first_tr_departure_tm = circuits.first_tr_departure_tm.map(time_tp).to_numpy()

    # Read the crew rotation parameters
    from params import sim_params
    
    # Build the Simulation args tuple
    args_sim = (
        args_circuits,
        n_start_trains,
        n_max_trains,
        first_tr_departure_tm,
        args_segments,
        args_stations,
        *tuple(map(lambda t : 60.0 * t, sim_params))
    )
    return args_sim, circuits, c_full.drop(columns= ['segm_code', 'segm_code_dir', 'segm_code_uni']), stations