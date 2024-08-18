import numpy as np
from numba import typeof, types, bool_, float64, int64
from numba.experimental import jitclass
from numba.typed import List

from .event_list import EventList, time_tp, event_tp
from .sim_objs import Train, Segment, Station, Circuit, idx_tp, pos_tp

circuit_tp = Circuit.class_type.instance_type
circuit_arr_tp = types.ListType(circuit_tp)
segm_tp = Segment.class_type.instance_type
segm_arr_tp = types.ListType(segm_tp)
st_tp = Station.class_type.instance_type
st_arr_tp = types.ListType(st_tp)
tr_tp = Train.class_type.instance_type
train_arr_tp = types.ListType(tr_tp)
reward_tp = float64
tr_wait_start_tp = types.Array(time_tp,1,'C')
cycle_stats_item_tp = typeof((idx_tp(0), idx_tp(0), np.array([time_tp(0)])))
arrival_dep_stats_tp = types.ListType(cycle_stats_item_tp)
snp_stats_item_tp = typeof((idx_tp(0), idx_tp(0), idx_tp(0), time_tp(0)))
snapshot_stats_tp = types.ListType(snp_stats_item_tp)
terminals_use_tp = types.Array(time_tp,1,'C')
state_tr_item_tp = typeof((idx_tp(0), pos_tp(0), bool_(0), idx_tp(0)))
graph_stats_item_tp = typeof((idx_tp(0), idx_tp(0), idx_tp(0), idx_tp(0), pos_tp(0), time_tp(0)))
graph_stats_tp = types.ListType(graph_stats_item_tp)

# TrainItinerarySim class attributes
spec = [
    ('events', EventList.class_type.instance_type), # List of next discrete events with their time and event code
    ('sim_tm', time_tp), # Simulation time
    ('reward', reward_tp), # Cumulative reward for this step
    ('tr_wait_start', tr_wait_start_tp), # List of waiting start times per train, to compute rewards
    ('circuits', circuit_arr_tp), # Circuit list
    ('segments', segm_arr_tp), # Segment list (non directed)
    ('stations', st_arr_tp), # Station list
    ('trains', train_arr_tp), # Train list
    ('terminals_use', terminals_use_tp), # Array containing the scheduled process completion time for each station
    ('arrival_stats', arrival_dep_stats_tp), # Acumulated statistics of train arrivals (circuit ID, train ID, arrival times)
    ('departure_stats', arrival_dep_stats_tp), # Acumulated statistics of train departures (circuit ID, train ID, departure times)
    ('snapshot_stats', snapshot_stats_tp), # Individual snapshots of train positions and times (circuit ID, train ID, timestamp, cycle completion ratio)
    ('graph_stats', graph_stats_tp), # Graph position statistics, as a list of tuples (circuit ID, train ID, graph ID, position, time)
    ('crew_rot_first_tm', time_tp), # Time of the first crew rotation event in minutes
    ('crew_rot_freq_tm', time_tp), # Frequency of the crew rotation event in minutes
    ('crew_rot_tm', time_tp), # Duration of the crew rotation event in minutes
    ('oth_wait_tm', pos_tp), # Wait time for the 2nd conflict train if not given priority
    ('dec_wait_tm', pos_tp), # Wait time for the 1st conflict train if not given priority
    ('decision_train_id', idx_tp), # ID of the train that is currently on hold waiting for an action
    ('other_train_id', idx_tp), # ID of the other train that competes for priority (lowest departure time)
    ('open_circuits_lo', event_tp), # Lower bound (not inclusive) for open circuit spawn event code
    ('acum_stats', bool_), # Whether to acumulate statistics of the train positions for plotting or debugging
]

@jitclass(spec)
class TrainItinerarySim:
    '''
    Main class to store the simulation state and implement the event logic.
    The constructor receives the parameters built in the input_preproc module, and also a random number generator for initizalizing the train positions.
    The simulation exposes a step and next_step methods, which are the interface for the RL loop (or any other optimization method for the meet decision problem).
    They return the reward accumulated during the step and the state vector representation of the operational state.
    The simulate method iterates performing the train movements until a decision must be made, or the predefined end time is reached.
    '''
    def __init__(self, rng, acum_stats, end_tm, arg_circuits,
                 n_start_trains, n_max_trains, first_tr_departure_tm,
                 arg_segments, arg_stations, first_crew_rot_hr, rot_freq_hr, rot_tm
                ):
        self.events = EventList()
        self.sim_tm = 0.0
        self.reward = 0.0
        self.crew_rot_first_tm = first_crew_rot_hr
        self.crew_rot_freq_tm = rot_freq_hr
        self.crew_rot_tm = rot_tm
        self.oth_wait_tm = 0.0
        self.dec_wait_tm = 0.0
        n_trains = int64(n_max_trains.sum())
        self.tr_wait_start = np.full(n_trains, time_tp(0.0))
        self.circuits = List.empty_list(circuit_tp, allocated= len(arg_circuits))
        for aci in arg_circuits:
            self.circuits.append(Circuit(*aci))
        self.segments = List.empty_list(segm_tp, allocated= len(arg_segments))
        for asm in arg_segments:
            self.segments.append(Segment(*asm))
        self.stations = List.empty_list(st_tp, allocated= len(arg_stations))
        for ast in arg_stations:
            self.stations.append(Station(*ast))
        self.trains = List.empty_list(tr_tp, allocated= n_trains)
        self.terminals_use = np.full(len(self.stations), time_tp(0.0))
        self.arrival_stats = List.empty_list(cycle_stats_item_tp)
        self.departure_stats = List.empty_list(cycle_stats_item_tp)
        self.snapshot_stats = List.empty_list(snp_stats_item_tp)
        self.graph_stats = List.empty_list(graph_stats_item_tp)
        self.decision_train_id = idx_tp(-1)
        self.other_train_id = idx_tp(-1)
        self.open_circuits_lo = event_tp(-len(self.circuits) - 1)
        self.acum_stats = acum_stats

        # Using rng to spawn trains randomly, but always during a journey, to make initialization easier. Repeat if the segment is already in use by another train.
        # Sort the circuit IDs by ascending number of segments, to allocate first the shorter circuits.
        circuits_idx_by_len = [idx_tp(p[1]) for p in sorted(zip([c.len for c in self.circuits], np.arange(len(self.circuits), dtype= idx_tp)))]
        total_added_trains = idx_tp(0)
        used_segms = np.full(len(self.segments), False, bool)
        for cidx in circuits_idx_by_len:
            circuit = self.circuits[cidx]
            # Train IDs for a particular circuit will always be ordered ascending
            circuit.train_ids = np.arange(int64(total_added_trains), int64(total_added_trains + n_max_trains[cidx]), dtype= idx_tp)
            # For each circuit, compute the individual travel times by segment
            travel_tm = np.empty_like(circuit.acum_entry_tm)
            travel_tm[0] = circuit.acum_entry_tm[0]
            travel_tm[1:] = circuit.acum_entry_tm[1:] - circuit.acum_exit_tm[:circuit.len-1]
            to_add = n_start_trains[cidx]
            rand_segm_idx = np.empty(0, dtype= idx_tp)
            while to_add > idx_tp(0):
                if np.all(used_segms[circuit.segment_ids]):
                    raise IndexError(f"Number of trains to place exceeds available segments in circuit {cidx}. The circuits are populated in this order: {circuits_idx_by_len}")
                # Place trains randomly in track segments
                candidates = rng.permutation(circuit.len)[:to_add]
                for k in candidates:
                    seg_id = circuit.segment_ids[k]
                    if not used_segms[seg_id]:
                        used_segms[seg_id] = True
                        to_add -= 1
                        rand_segm_idx = np.append(rand_segm_idx, idx_tp(k))
            # Directed segment IDs to be used are sorted in descending order, so that trains are sorted by cycle position
            rand_segm_idx = -np.sort(-rand_segm_idx)
            # Add each train
            for k in rand_segm_idx:
                seg_id = circuit.segment_ids[k]
                origin_st_id = circuit.dest_st_id[k-1] if k > idx_tp(0) else circuit.origin_st_id
                # Generate random progress, compute remaining time
                next_arrival_tm = time_tp(rng.uniform(1e-5, travel_tm[k]-1e-5))
                current_shift = -circuit.acum_entry_tm[k] + next_arrival_tm
                est_arrival_tm = circuit.acum_entry_tm + current_shift
                est_depart_tm = circuit.acum_exit_tm + current_shift
                self.trains.append(Train(cidx, k, k,
                                        seg_id, origin_st_id,
                                        circuit.dest_st_id[k],
                                        est_arrival_tm, est_depart_tm))
                self.segments[seg_id].occupy(total_added_trains, self.acum_stats, self.sim_tm)
                total_added_trains += 1
                self.events.push(next_arrival_tm, event_tp(total_added_trains)) # Event code is shifted by 1 with respect to train IDs
            # For open circuits, non spawned trains must be added as out of the network
            if circuit.next_spawn != idx_tp(-1):
                # Add next spawn time to events
                nxt_spwn_tm = first_tr_departure_tm[cidx]
                prv_rot_tm = ((nxt_spwn_tm - self.crew_rot_first_tm) // self.crew_rot_freq_tm) * self.crew_rot_freq_tm + self.crew_rot_first_tm
                nxt_spwn_tm += (self.crew_rot_tm + 1e-5) if (nxt_spwn_tm - prv_rot_tm) <= self.crew_rot_tm else 0.0 # If the spawn was to happen during crew rotation, shift forward
                nxt_spwn_tm -= self.crew_rot_tm * (1.0 + (nxt_spwn_tm - self.crew_rot_first_tm) // self.crew_rot_freq_tm) # Crew rotation events will add this time back again
                self.events.push(nxt_spwn_tm, event_tp(-cidx-1))
                # Initialize trains            
                est_depart_tm = circuit.acum_exit_tm - circuit.cycle_tm
                next_depart_idx = circuit.len - 1
                seg_id = circuit.segment_ids[0]
                origin_st_id = circuit.dest_st_id[-1]
                for k in range(n_max_trains[cidx] - n_start_trains[cidx]):
                    self.trains.append(Train(cidx, idx_tp(0), next_depart_idx,
                                            seg_id, origin_st_id,
                                            idx_tp(-1),
                                            circuit.acum_entry_tm.copy(), est_depart_tm.copy()))
                    self.trains[total_added_trains].is_waiting = True
                    total_added_trains += 1
        
        # Add first crew rotation and simulation end event to the list
        self.events.push(time_tp(self.crew_rot_first_tm), event_tp(0)) # First shift rotation event
        self.events.push(time_tp(end_tm), self.open_circuits_lo) # Simulation end time

    def register_moving_trains_loc(self):
        '''
        Save the position of all moving trains, to show in the graph as stopped when there is a crew rotation
        '''
        for segm in self.segments:
            tr_id = segm.train_id
            if tr_id != idx_tp(-1):
                train = self.trains[tr_id]
                train_arr_tm = train.est_arrival_tm[train.next_arrival_idx]
                curr_tm = self.sim_tm
                progress = pos_tp(1.0 - (train_arr_tm - curr_tm) / (train_arr_tm - segm.util_stats[-1][0]))
                self.graph_stats.append((train.circuit_id, tr_id, train.origin_st_id, train.dest_st_id, progress, curr_tm))
                self.graph_stats.append((train.circuit_id, tr_id, train.origin_st_id, train.dest_st_id, progress, curr_tm + self.crew_rot_tm))
    
    def update_single_reward(self, tr_id):
        '''
        Update the reward from a single train that has stopped waiting
        '''
        self.reward -= self.sim_tm - self.tr_wait_start[tr_id]
        self.tr_wait_start[tr_id] = 0.0
        
    def update_all_rewards(self):
        '''
        Compute the partial waiting times up to this point for trains that are still waiting
        '''
        local_reward = reward_tp(0.0)
        current_tm = self.sim_tm
        for k in range(len(self.tr_wait_start)):
            if self.tr_wait_start[k] != 0.0:
                local_reward -= current_tm - self.tr_wait_start[k]
                self.tr_wait_start[k] = current_tm
        self.reward += local_reward
    
    def unfreeze_train(self, tr_id, extra = time_tp(0.0)):
        '''
        Mark train as not waiting. Add waiting time to reward. Update estimated arrival and departure times after pause.
        '''
        train = self.trains[tr_id]
        train.is_waiting = False
        self.update_single_reward(tr_id)
        shift_t = self.sim_tm - train.est_depart_tm[train.next_depart_idx] + extra
        train.est_arrival_tm[train.next_arrival_idx:] += shift_t
        train.est_depart_tm[train.next_depart_idx:] += shift_t  
    
    def move_train(self, tr_id):
        '''
        Train has been authorized to move. Clear the station, occupy the segment, signal the train waiting in the previous station. Schedule arrival.
        '''
        train = self.trains[tr_id]
        circuit = self.circuits[train.circuit_id]
        if train.is_waiting:
            self.unfreeze_train(tr_id)
            
        # Advance next departure index
        train.next_depart_idx += 1
        if train.next_depart_idx == circuit.len:
            if self.acum_stats:
                if circuit.next_spawn == idx_tp(-1): # Closed circuits report stats when departing from the last station (origin of the next cycle)
                    self.departure_stats.append((train.circuit_id, tr_id, train.est_depart_tm[:train.next_depart_idx].copy()))
                    cycle_st = train.cycle_stats[-1][0]
                    train.cycle_stats[-1] = (cycle_st, self.sim_tm - cycle_st)
                    train.cycle_stats.append((self.sim_tm, np.nan))
                else: # For open circuits, departure from origin is done on spawn, so all previous departure times are fake. Only the current time is registered.
                    self.snapshot_stats.append((train.circuit_id, tr_id, train.current_st_id, self.sim_tm))
            train.next_depart_idx = 0
            train.est_depart_tm[:] = circuit.acum_exit_tm + self.sim_tm

        # Notify station and segment about changes
        tr_behind = self.stations[train.current_st_id].depart(train.origin_st_id)
        self.segments[train.tgt_segment].occupy(tr_id, self.acum_stats, self.sim_tm)
        train.depart()

        # Make the trian behind check for permission to move by adding an event at the same timestamp
        if tr_behind != idx_tp(-1):
            self.unfreeze_train(tr_behind)
            self.events.push(self.sim_tm, event_tp(tr_behind+1))

        # Add arrival to event list
        self.events.push(train.est_arrival_tm[train.next_arrival_idx], event_tp(tr_id+1))
    
    def wait_for_other(self, tr_id, other_tr_id):
        '''
        Stop train until the other clears the track and sends a signal to move
        '''
        train = self.trains[tr_id]
        train.is_waiting = True
        self.trains[other_tr_id].add_waiter(train.current_st_id, tr_id)
        self.tr_wait_start[tr_id] = self.sim_tm

    def wait_for_station(self, tr_id, st_id):
        '''
        Stop train until the station has capacity to receive one more train. Signal other trains to move in special cases.
        '''
        train = self.trains[tr_id]
        train.is_waiting = True
        self.stations[st_id].add_waiter(train.current_st_id, tr_id)
        self.tr_wait_start[tr_id] = self.sim_tm

        # If a train is waiting for this one in a far away station, give it permission to move, to avoid deadlocks
        other_tr_ids = train.signal_far_waiters()
        for tr_id_oth in other_tr_ids:
            self.unfreeze_train(tr_id_oth)
            self.events.push(self.sim_tm, event_tp(tr_id_oth+1))

        # If a train is waiting for this one in the next station, give it permission to move, to avoid deadlocks
        was_opposite, tr_id_oth, other_orig_wl, other_dest_wl = train.transfer_wait_lists()
        if tr_id_oth != idx_tp(-1):
            self.trains[tr_id_oth].receive_wait_lists(other_orig_wl, other_dest_wl)
            self.move_train(tr_id_oth)
    
    def request_authorization(self, tr_id):
        '''
        Check if there is another train already using the segment.
        Check if the next station has capacity for more trains in this direction.
        Loop over circuits and trains looking when will they enter the segment. Exclude trains that come from behind (have same origin).
        Check if the incoming train will enter the segment before the requesting train will arrive at the next station (conflict found).
        Apply priorities according to operator.
        If there is a conflict between trains with the same priority, get the opposite train which is closest. Inform that a decision must be made.
        '''
        train = self.trains[tr_id]
        tgt_segment = self.segments[train.tgt_segment]
        tr_other_id = tgt_segment.train_id
        if tr_other_id == idx_tp(-1): # No train in target segment
            circuit = self.circuits[train.circuit_id]
            tgt_station_id = circuit.dest_st_id[train.next_arrival_idx]
            tgt_station = self.stations[tgt_station_id]
            current_station_id = train.current_st_id
            if tgt_station.has_cap(current_station_id): # Station has capacity to receive another train in this direction
                if len(train.dest_wait_list) > 0 or len(train.far_wait_list) > 0: # If the train is being waited up ahead, move, to avoid deadlocks
                    self.move_train(tr_id)
                    return True, idx_tp(-1)
                min_other_depart_tm = time_tp(np.inf)
                min_other_tr_id = idx_tp(-1)
                will_wait = False
                tgt_arrival_tm = train.est_arrival_tm[train.next_arrival_idx]
                for circuit_id_other, it_idxs in tgt_segment.circuits: # Loop over circuits and itinerary indexes that pass through this segment
                    circuit_other = self.circuits[circuit_id_other]
                    # If both trains start from the same station, the current one gets priority because it arrived first. Discard these indeces.
                    valid_it_idxs = [i for i in it_idxs if circuit_other.dest_st_id[i] != current_station_id]
                    if len(valid_it_idxs) > 0:
                        # Determine if the other circuit has more priority over the segment
                        has_less_priority = not tgt_segment.priority_circuits[train.circuit_id] and tgt_segment.priority_circuits[circuit_id_other]
                        is_open_circuit = circuit_other.next_spawn != idx_tp(-1)
                        # Iterate over the other circuit's trains
                        for tr_id_other in circuit_other.train_ids:
                            train_other = self.trains[tr_id_other]
                            if not train_other.is_waiting and tr_id != tr_id_other: # Waiting trains never impose waiting on others, to avoid gridlock
                                for it_idx in valid_it_idxs:
                                    depart_idx_offset = idx_tp(0) if is_open_circuit or (it_idx >= train_other.next_depart_idx) else circuit_other.len
                                    other_depart_tm = train_other.est_depart_tm[it_idx + depart_idx_offset]
                                    if tgt_arrival_tm > other_depart_tm and other_depart_tm + 1e-6 > self.sim_tm: # There is a conflict between the trains to use the segment it_idx
                                        will_wait = max(will_wait, has_less_priority)
                                        if min_other_depart_tm > other_depart_tm: # The train that will arrive sooner is labeled as the opposite, for the RL model
                                            min_other_depart_tm = other_depart_tm
                                            min_other_tr_id = tr_id_other
                                            # Compute wait times for each train, as a feature for the RL model
                                            self.oth_wait_tm = tgt_arrival_tm - other_depart_tm
                                            if it_idx != idx_tp(circuit_other.len-1):
                                                oth_travel_tm = circuit_other.acum_entry_tm[it_idx+1] - circuit_other.acum_exit_tm[it_idx]
                                            else:
                                                oth_travel_tm = circuit_other.acum_entry_tm[0]
                                            self.dec_wait_tm = other_depart_tm + oth_travel_tm - self.sim_tm
                                        
                if will_wait: # Forced wait by priority
                    self.wait_for_other(tr_id, min_other_tr_id)
                elif min_other_tr_id != idx_tp(-1): # Stop and wait for decision
                    return False, min_other_tr_id
                else: # No blocking trains, continue moving
                    self.move_train(tr_id)
            else: # Target station is occupied, notify the station as waiter
                self.wait_for_station(tr_id, tgt_station_id)
        else: # Segment is occupied, notify the occupying train as waiter
            self.wait_for_other(tr_id, tr_other_id)
        return True, idx_tp(-1)
    
    def next_process(self, tr_id):
        '''
        Determine if the train is arriving or trying to leave a station. Detect meet. Signal waiting train to move. 
        Stay for remaining process at stationor request authorization to continue.
        '''
        train = self.trains[tr_id]
        circuit = self.circuits[train.circuit_id]
        if train.next_arrival_idx == train.next_depart_idx: # Train is arriving
            # Notify station and segment about changes
            self.stations[train.dest_st_id].arrive(train.origin_st_id, tr_id)
            self.segments[train.tgt_segment].free(self.acum_stats, self.sim_tm)

            # If there are trains waiting to use the segment, give the first a signal
            next_segm_idx = train.next_depart_idx + 1 if train.next_depart_idx < (circuit.len - 1) else idx_tp(0)
            was_opposite, other_idx, other_orig_wl, other_dest_wl = train.arrive(circuit.dest_st_id[next_segm_idx])

            own_switch_tm = time_tp(0.0)
            if other_idx != idx_tp(-1):
                self.trains[other_idx].receive_wait_lists(other_orig_wl, other_dest_wl)
                moved = False
                if was_opposite: # If there was a meet, determine if there is delay due to switch process. Schedule movements based on this
                    own_switch_tm = circuit.passer_switch_tm[train.next_depart_idx]
                    if own_switch_tm != 0.0:
                        train.est_depart_tm[train.next_depart_idx:] += own_switch_tm
                        train.est_arrival_tm[train.next_arrival_idx+1:] += own_switch_tm
                    other_switch_tm = circuit.waiter_switch_tm[train.next_depart_idx]
                    self.reward -= own_switch_tm + other_switch_tm
                    if other_switch_tm == 0.0:
                        moved = True
                        self.move_train(other_idx)
                    elif self.acum_stats:
                        self.segments[train.tgt_segment].add_util(other_switch_tm)
                else:
                    other_switch_tm = time_tp(0.0)
                if not moved:
                    self.unfreeze_train(other_idx, extra= other_switch_tm)
                    self.events.push(self.sim_tm + other_switch_tm, event_tp(other_idx+1))
            
            # Advance next arrival index
            train.next_arrival_idx += 1
            if train.next_arrival_idx == circuit.len:
                if self.acum_stats:
                    self.arrival_stats.append((train.circuit_id, tr_id, train.est_arrival_tm.copy()))
                # If circuit is open, remove the train from the network and signal waiting trains
                if circuit.next_spawn != idx_tp(-1):
                    tr_behind = self.stations[train.current_st_id].depart(train.origin_st_id)
                    train.is_waiting = True
                    train.dest_st_id = -1                    
                    if self.acum_stats: # Arrival at the last station marks cycle completion for open circuits
                        self.departure_stats.append((train.circuit_id, tr_id, train.est_depart_tm[:circuit.len].copy()))
                        cycle_st = train.cycle_stats[-1][0]
                        train.cycle_stats[-1] = (cycle_st, self.sim_tm - cycle_st)
                        # Register train position at dummy station -1 to easily detect cycle end in post-processing
                        self.snapshot_stats.append((train.circuit_id, tr_id, idx_tp(-1), self.sim_tm+1e-6))
                    # Make the trian behind check for permission to move by adding an event at the same timestamp
                    if tr_behind != idx_tp(-1):
                        self.unfreeze_train(tr_behind)
                        self.events.push(self.sim_tm, event_tp(tr_behind+1))
                    return True, idx_tp(-1)
                else:
                    train.next_arrival_idx = 0
                    train.est_arrival_tm[:] = circuit.acum_entry_tm + train.est_depart_tm[train.next_depart_idx]
            # Get new target segment ID
            train.tgt_segment = circuit.segment_ids[train.next_arrival_idx]

            # Check if the train has to stay for any process in the station
            has_terminal_process = circuit.acum_exit_tm[train.next_depart_idx] - circuit.acum_entry_tm[train.next_depart_idx] - 1e-6 > 0.0
            next_depart_tm = train.est_depart_tm[train.next_depart_idx]
            has_any_process = next_depart_tm - self.sim_tm - 1e-6 > 0.0
        
            if has_terminal_process:
                st_process_queue_tm = self.terminals_use[train.current_st_id] - self.sim_tm - own_switch_tm
                if st_process_queue_tm > 0.0:
                    train.est_depart_tm[train.next_depart_idx:] += st_process_queue_tm
                    train.est_arrival_tm[train.next_arrival_idx:] += st_process_queue_tm
                    next_depart_tm += st_process_queue_tm
                    self.reward -= st_process_queue_tm
                self.terminals_use[train.current_st_id] = next_depart_tm + circuit.cooldown_tm[train.next_depart_idx]
            if has_any_process:
                self.events.push(next_depart_tm, event_tp(tr_id+1))
                return True, idx_tp(-1)
        # Request to move
        return self.request_authorization(tr_id)
    
    def spawn(self, circuit_id):
        '''
        Spawn a train from an open circuit
        1. Get starting segment and station. Initialize train attributes.
        2. Set station as occupied
        '''
        circuit = self.circuits[circuit_id]
        train_id = circuit.train_ids[circuit.next_spawn]
        train = self.trains[train_id]
        # If the train has left the network, it can be spawned and move
        if train.dest_st_id == idx_tp(-1):
            if circuit.next_spawn != idx_tp(len(circuit.train_ids)-1):
                circuit.next_spawn += 1
            else:
                circuit.next_spawn = 0
            # Initialize train in origin
            train.is_waiting = False
            train.next_arrival_idx = 0
            train.next_depart_idx = circuit.len - 1
            train.tgt_segment = circuit.segment_ids[0]
            train.origin_st_id = circuit.dest_st_id[-1]
            train.current_st_id = circuit.origin_st_id
            train.dest_st_id = circuit.dest_st_id[0]
            train.est_arrival_tm[:] = circuit.acum_entry_tm + self.sim_tm
            train.est_depart_tm[:] = circuit.acum_exit_tm + self.sim_tm - circuit.cycle_tm
            # Signal arrival and request authorization
            self.stations[train.current_st_id].arrive(train.origin_st_id, train_id)
            if self.acum_stats:
                self.snapshot_stats.append((circuit_id, train_id, train.current_st_id, self.sim_tm))
                train.cycle_stats.append((self.sim_tm, np.nan))
            no_action_needed, other_train_id = self.request_authorization(train_id)
            
        else:
            no_action_needed, other_train_id = True, idx_tp(-1)

        # Add next spawn to events
        nxt_spwn_tm = self.sim_tm + circuit.frequency
        prv_rot_tm = ((nxt_spwn_tm - self.crew_rot_first_tm) // self.crew_rot_freq_tm) * self.crew_rot_freq_tm + self.crew_rot_first_tm
        nxt_rot_tm = ((self.sim_tm - self.crew_rot_first_tm) // self.crew_rot_freq_tm + 1.0) * self.crew_rot_freq_tm + self.crew_rot_first_tm
        # If the spawn was going to happen during crew rotation, shift it forward to force the roration to happen before and displace it
        nxt_spwn_tm += (self.crew_rot_tm + 1e-5) if (nxt_spwn_tm - prv_rot_tm) <= self.crew_rot_tm else 0.0
        # Crew rotation events will add this time back again
        nxt_spwn_tm -= self.crew_rot_tm * (1.0 + (nxt_spwn_tm - nxt_rot_tm) // self.crew_rot_freq_tm) 
        self.events.push(nxt_spwn_tm, event_tp(-circuit_id-1))
        return no_action_needed, train_id, other_train_id
    
    def crew_rotation(self):
        '''
        Stop operations during crew shift rotation:
            1. All arrival and departure times are shifted forward
            2. Times in the event list are shifted forward
            3. All waiting durations are increased
            4. Terminal station availability time is shifted forward
        Add new crew rotation event.
        '''
        if self.acum_stats:
            self.register_moving_trains_loc()
        delta_t = self.crew_rot_tm
        self.events.sleep(delta_t)
        for train in self.trains:
            if not train.is_waiting:
                train.est_arrival_tm[train.next_arrival_idx:] += delta_t
                train.est_depart_tm[train.next_depart_idx:] += delta_t
        for k in range(len(self.tr_wait_start)):
            if self.tr_wait_start[k] != 0.0:
                self.tr_wait_start[k] += delta_t
        self.terminals_use += delta_t
        self.events.push(self.sim_tm + self.crew_rot_freq_tm, event_tp(0))
    
    def simulate(self):
        '''
        Execute the simulation in loop, taking events from the events list and advancing time, until a decision must be made, or the predefined end time is reached.
        The codes for each event action are:
            >0: Next process of a train (arrival or departure). Train ID = event_code - 1
            =0: Crew rotation
            <0: Spawn of a train from open circuit (except for event_code = self.open_circuits_lo). Circuit ID = -event_code - 1
            =self.open_circuits_lo: simulation end
        '''
        no_action_needed = True
        other_train_id = idx_tp(-1)
        while(no_action_needed):
            time, event_code = self.events.peek()
            self.sim_tm = time
            if event_code > event_tp(0): # Next process of a train
                no_action_needed, other_train_id = self.next_process(idx_tp(event_code-1))
                self.decision_train_id = idx_tp(event_code-1)
            elif event_code < event_tp(0) and event_code > self.open_circuits_lo: # Spawn of a train from an open circuit
                no_action_needed, tr_id, other_train_id = self.spawn(idx_tp(-event_code-1))
                self.decision_train_id = tr_id
            elif event_code == event_tp(0): # Crew rotation
                self.crew_rotation()
            else: # Simulation end (event_code = self.open_circuits_lo)
                break
        self.other_train_id = other_train_id
        return not no_action_needed

    def get_state(self, not_ended):
        '''
        Compute state vector representation for the actor-critic network
        '''
        NT = len(self.trains)
        fixed_feats = 12
        size = fixed_feats + 2*NT
        state = np.empty(size, dtype= np.float32)
        # loc 11+2*NT: has ended
        state[size-1] = not not_ended
        if not_ended:
            # loc 0: Segment ID (non-directed)
            tr_dec = self.trains[self.decision_train_id]
            state[0] = tr_dec.tgt_segment
            # loc 1: Whether the current station for the decision train has lower station ID than the destination station
            state[1] = tr_dec.current_st_id < tr_dec.dest_st_id
            # loc 2-3: Circuit ID for both trains
            state[2] = tr_dec.circuit_id
            tr_oth = self.trains[self.other_train_id]
            state[3] = tr_oth.circuit_id
            # loc 11-(10+NT): Compute cycle time completion ratios. Sort trains by position for each circuit.
            # loc (11+NT)-(10+2*NT): Whether each train is waiting (using the same sorting by position)
            # Make a list of (circuit_id, cycle_position, is_waiting, train_id)
            tr_pos_ls = List.empty_list(state_tr_item_tp, allocated= NT)
            for cid, c in enumerate(self.circuits):
                for tr_id in c.train_ids:
                    train = self.trains[tr_id]
                    cycle_pos = pos_tp(2.0)
                    if train.dest_st_id != idx_tp(-1):
                        cycle_pos = pos_tp((train.est_depart_tm[c.len-1] - train.est_depart_tm[train.next_depart_idx]) / c.cycle_tm)
                    tr_pos_ls.append((idx_tp(cid), cycle_pos, train.is_waiting, tr_id))
            # Sort the list to get the trains sorted by circuit and cycle position
            tr_pos_ls_sort = sorted(tr_pos_ls)
            dec_tr_pos_id, oth_tr_pos_id, last_circuit, circuit_pos, dec_circuit_pos, oth_circuit_pos = 0, 0, idx_tp(-1), idx_tp(0), idx_tp(0), idx_tp(0)
            for i, tup in enumerate(tr_pos_ls_sort):
                state[fixed_feats-1+i] = tup[1]
                state[fixed_feats-1+i+NT] = tup[2]
                if last_circuit != tup[0]:
                    last_circuit = tup[0]
                    circuit_pos = idx_tp(0)
                else:
                    circuit_pos += 1
                if tup[3] == self.decision_train_id:
                    dec_tr_pos_id = i
                    dec_circuit_pos = circuit_pos
                elif tup[3] == self.other_train_id:
                    oth_tr_pos_id = i
                    oth_circuit_pos = circuit_pos
            # loc 4-5: Sorted position in complete train list for both the decision train and other train
            state[4] = dec_tr_pos_id
            state[5] = oth_tr_pos_id
            # loc 6: Decision train waiting time - other train waiting time in hours
            state[6] = (self.dec_wait_tm - self.oth_wait_tm) / 60.0
            # loc 7: Decision train cycle completion - previous train cycle completion - theoretical frequency (1/Ntc)
            dec_circuit = self.circuits[tr_dec.circuit_id]
            dec_ntc = idx_tp(len(dec_circuit.train_ids))
            freq = 1.0/dec_ntc
            if dec_circuit_pos:
                state[7] = tr_pos_ls_sort[dec_tr_pos_id][1] - tr_pos_ls_sort[dec_tr_pos_id - 1][1] - freq
            else:
                state[7] = tr_pos_ls_sort[dec_tr_pos_id][1] - tr_pos_ls_sort[dec_tr_pos_id + dec_ntc - 1][1] + 1.0 - freq
            # loc 8: Next train cycle completion - decision train cycle completion - theoretical frequency (1/Ntc)
            if dec_circuit_pos != idx_tp(dec_ntc-1):
                state[8] = tr_pos_ls_sort[dec_tr_pos_id + 1][1] - tr_pos_ls_sort[dec_tr_pos_id][1] - freq
            else:
                state[8] = tr_pos_ls_sort[dec_tr_pos_id - dec_ntc + 1][1] - tr_pos_ls_sort[dec_tr_pos_id][1] + 1.0 - freq
            # loc 9: Other train cycle completion - previous train cycle completion - theoretical frequency (1/Ntc)
            oth_circuit = self.circuits[tr_oth.circuit_id]
            oth_ntc = idx_tp(len(oth_circuit.train_ids))
            freq = 1.0/oth_ntc
            if oth_circuit_pos:
                state[9] = tr_pos_ls_sort[oth_tr_pos_id][1] - tr_pos_ls_sort[oth_tr_pos_id - 1][1] - freq
            else:
                state[9] = tr_pos_ls_sort[oth_tr_pos_id][1] - tr_pos_ls_sort[oth_tr_pos_id + oth_ntc - 1][1] + 1.0 - freq
            # loc 10: Next train cycle completion - other train cycle completion - theoretical frequency (1/Ntc)
            if oth_circuit_pos != idx_tp(oth_ntc-1):
                state[10] = tr_pos_ls_sort[oth_tr_pos_id + 1][1] - tr_pos_ls_sort[oth_tr_pos_id][1] - freq
            else:
                state[10] = tr_pos_ls_sort[oth_tr_pos_id - oth_ntc + 1][1] - tr_pos_ls_sort[oth_tr_pos_id][1] + 1.0 - freq
            # loc 11: time of day, from 0 to 1 (24 hours). Currently not used
            #state[11] = (self.sim_tm % 24.0) / 24.0
        return state

    def first_step(self, compute_state : bool = True):
        '''
        Advance new simulation up to the point where a first action is needed
        Returns no reward as there is no previous action to reward.
        '''
        not_ended = self.simulate()

        self.update_all_rewards()
        if compute_state:
            state = self.get_state(not_ended)
        else:
            state = self.get_state(False)

        return state
    
    def step(self, has_priority : bool, compute_state : bool = True): 
        '''
        Receive action to execute in the current state, return next state and reward.
        The simulation had stopped waiting for an action (priority to decision train [True] or the other train [False]).
        '''
        self.reward = reward_tp(0.0)
        
        if has_priority:
            self.move_train(self.decision_train_id)
        else:
            self.wait_for_other(self.decision_train_id, self.other_train_id)

        not_ended = self.simulate()

        self.update_all_rewards()
        if compute_state:
            state = self.get_state(not_ended)
        else:
            state = self.get_state(False)

        return (self.reward, state)
