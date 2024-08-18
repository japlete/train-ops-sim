import numpy as np
from numba import typeof, types, int64, uint8, bool_, float32
from numba.typed import List
from numba.experimental import jitclass

from .event_list import time_tp

idx_tp = int64
small_counter = uint8
pos_tp = float32
time_array_tp = types.Array(time_tp,1,'C')
idx_list_tp = types.ListType(idx_tp)
idx_arr_tp = types.Array(idx_tp,1,'C')
small_count_arr = types.Array(small_counter,1,'C')
idx_tuple_tp = typeof((idx_tp(0), idx_tp(0)))
list_idx_idx_tp = types.ListType(idx_tuple_tp)
bool_arr_tp = types.Array(bool_,1,'C')
list_idx_idxarr_tp = types.ListType(types.Tuple((idx_tp, types.Array(idx_tp,1,'C'))))
dict_idx_idxarr_tp = types.DictType(idx_tp, types.ListType(idx_tp))
time_tuple_tp = typeof((time_tp(0), time_tp(0)))
list_tm_tm_tp = types.ListType(time_tuple_tp)

# Train class attributes
spec_train = [
    ('circuit_id', idx_tp), # Circuit ID of which this train is part of
    ('is_waiting', bool_), # Is the train waiting or executing a process
    ('next_arrival_idx', idx_tp), # Directed Segment ID for the next arrival (at destination station)
    ('next_depart_idx', idx_tp), # Directed Segment ID for the next departure (at destination station)
    ('tgt_segment', idx_tp), # Target segment ID (non directed), currently traversing, or next (if already arrived)
    ('origin_st_id', idx_tp), # Station ID of origin for the current segment
    ('current_st_id', idx_tp), # Station ID of the current station if the train isn't moving
    ('dest_st_id', idx_tp), # Station ID of destination for the current segment
    ('est_arrival_tm', time_array_tp), # Estimated time of arrival for each destination station
    ('est_depart_tm', time_array_tp), # Estimated time of departure for each destination station for the current and next cycle (for closed circuit)
    ('dest_wait_list', idx_list_tp), # Train ID of trains that are waiting for the segment to clear at destination
    ('orig_wait_list', idx_list_tp), # Train ID of trains that are waiting for the segment to clear at origin
    ('far_wait_list', dict_idx_idxarr_tp), # Train ID of trains that are waiting for this train in a different segment
    ('cycle_stats', list_tm_tm_tp), # Cycle time statistics, as a list of tuples (start time, duration)
]

@jitclass(spec_train)
class Train:
    '''
    A train stores information about its position in the network, IDs of other trains that are waiting for it, and its cycle time statistics.
    '''
    def __init__(self, circuit_id, next_arrival_idx, next_depart_idx, tgt_segment,
                 origin_st_id, dest_st_id, est_arrival_tm, est_depart_tm
                ):
        self.circuit_id = circuit_id
        self.is_waiting = False
        self.next_arrival_idx = next_arrival_idx
        self.next_depart_idx = next_depart_idx
        self.tgt_segment = tgt_segment
        self.origin_st_id = origin_st_id
        self.current_st_id = idx_tp(-1)
        self.dest_st_id = dest_st_id
        self.est_arrival_tm = est_arrival_tm
        self.est_depart_tm = est_depart_tm
        self.dest_wait_list = List.empty_list(idx_tp)
        self.orig_wait_list = List.empty_list(idx_tp)
        self.far_wait_list = {idx_tp(0) : List.empty_list(idx_tp)}
        del self.far_wait_list[0]
        self.cycle_stats = List.empty_list(time_tuple_tp)
        self.cycle_stats.append((np.nan, np.nan))
 
    def depart(self):
        '''
        Update the origin, current and destination station IDs
        '''
        self.origin_st_id = self.current_st_id
        self.current_st_id = idx_tp(-1)
        
    def add_waiter(self, waiter_orig_st_id, waiter_id):
        '''
        Add train to the corresponding waiting list (in destination station, at origin or further away)
        '''
        if self.dest_st_id == waiter_orig_st_id:
            self.dest_wait_list.append(waiter_id)
        elif self.origin_st_id == waiter_orig_st_id:
            self.orig_wait_list.append(waiter_id)
        elif waiter_orig_st_id not in self.far_wait_list:
            self.far_wait_list[waiter_orig_st_id] = List([waiter_id])
        else:
            self.far_wait_list[waiter_orig_st_id].append(waiter_id)

    def receive_wait_lists(self, orig_wait_list, dest_wait_list):
        '''
        Receive the waiting list from another train that has left the segment.
        '''
        self.dest_wait_list.extend(dest_wait_list)
        self.orig_wait_list.extend(orig_wait_list)

    def transfer_wait_lists(self):
        '''
        Clear origin and destination waiting lists after leaving a segment. Return a flag if there is a meet, the train ID of the other train,
        and the remainder of the waiting lists after removing this train. The lists are flipped in the return tuple depending on the direction of the receiving train.
        '''
        if len(self.dest_wait_list) > 0:
            ret = True, self.dest_wait_list[0], self.dest_wait_list[1:].copy(), self.orig_wait_list.copy()
        elif len(self.orig_wait_list) > 0:
            ret = False, self.orig_wait_list[0], self.orig_wait_list[1:].copy(), self.dest_wait_list.copy()
        else:
            ret = False, idx_tp(-1), self.dest_wait_list.copy(), self.orig_wait_list.copy()
        self.dest_wait_list.clear()
        self.orig_wait_list.clear()
        return ret

    def arrive(self, new_dest_st_id):
        '''
        Update the current and destination station IDs after arriving at the destination station. Return the wait lists to be transfered.
        If the new destination station is in the far_wait_list, set these waiting trains to the new destination wait list.
        '''
        was_opposite, other_idx, other_orig_wl, other_dest_wl = self.transfer_wait_lists()
        self.current_st_id = self.dest_st_id
        self.dest_st_id = new_dest_st_id
        if new_dest_st_id in self.far_wait_list:
            self.dest_wait_list = self.far_wait_list.pop(new_dest_st_id)
        return was_opposite, other_idx, other_orig_wl, other_dest_wl

    def signal_far_waiters(self):
        '''
        Retrieve the first train in the far_wait_list for each origin station, and remove the train from the list.
        '''
        n_stations = len(self.far_wait_list)
        first_per_station = np.empty(n_stations, dtype= idx_tp)
        if n_stations > 0:
            added = 0
            keys_to_del = List.empty_list(idx_tp, allocated= n_stations)
            for other_orig_st_id, waiters in self.far_wait_list.items():
                first_per_station[added] = waiters.pop(0)
                if len(waiters) == 0:
                    keys_to_del.append(other_orig_st_id)
                added += 1
            for k in keys_to_del:
                del self.far_wait_list[k]
        return first_per_station
  
# Station class attributes
spec_station = [
    ('origin_st_id', idx_arr_tp), # List of stations that are origins of segments to this one
    ('st_cap', small_counter), # Number of trains that fit into this station, coming from each origin
    ('origin_st_use', small_count_arr), # Number of trains currently using each track
    ('wait_list', list_idx_idx_tp), # List of origin station IDs and train IDs that are waiting for a signal to move to this station when there is capacity. FIFO
]

@jitclass(spec_station)
class Station:
    '''
    The Station stores information about the number of trains that can fit into a station per track, the number of trains currently using each track, 
    and the waiting list to move to this station from each origin station, if the corresponding track is full.
    '''
    def __init__(self, origin_st_id, st_cap):
        self.origin_st_id = origin_st_id
        self.st_cap = small_counter(st_cap)
        self.origin_st_use = np.zeros(len(self.origin_st_id), dtype= small_counter)
        self.wait_list = List.empty_list(idx_tuple_tp)

    def has_cap(self, origin_id : idx_tp):
        '''
        Determine if the track reserved for trains from the given origin station has capacity for more.
        '''
        for k in range(len(self.origin_st_id)):
            if origin_id == self.origin_st_id[k]:
                return self.origin_st_use[k] < self.st_cap
        return False

    def add_waiter(self, origin_id : idx_tp, train_id : idx_tp):
        '''
        Add a train to the waiting list for the given origin station.
        '''
        self.wait_list.append((origin_id, train_id))

    def depart(self, origin_id : idx_tp):
        '''
        When a train departs, update the capacity of the track reserved for the origin station from where it came.
        If there is a train waiting from this origin station, return the train ID.
        '''
        for k in range(len(self.origin_st_id)):
            if origin_id == self.origin_st_id[k]:
                self.origin_st_use[k] -= 1
                break
        for k in range(len(self.wait_list)):
            if self.wait_list[k][0] == origin_id:
                return self.wait_list.pop(k)[1]
        return idx_tp(-1)

    def arrive(self, origin_id : idx_tp, train_id : idx_tp):
        '''
        When a train arrives, update the capacity of the track reserved for the origin station from where it came.
        Remove the train from the waiting list if it's there.
        '''
        for k in range(len(self.origin_st_id)):
            if origin_id == self.origin_st_id[k]:
                self.origin_st_use[k] += 1
                break
        remove_tup = (origin_id, train_id)
        for k in range(len(self.wait_list)):
            if self.wait_list[k] == remove_tup:
                del self.wait_list[k]
                break

# Segment class attributes
spec_segment = [
    ('train_id', idx_tp), # ID of train that is currently traversing the segment
    ('circuits', list_idx_idxarr_tp), # list of circuit IDs that contain this segment, with its positions in the itinerary
    ('priority_circuits', bool_arr_tp), # Boolean flag indicating if a circuit has priority over this segment
    ('util_stats', list_tm_tm_tp), # Segment utilization statistics, as a list of tuples (start time, duration)
]

@jitclass(spec_segment)
class Segment:
    '''
    The Segment stores the ID of the train that is currently traversing the segment, the circuits which have priority over the segment,
    utilization factor statistics and the list of circuit IDs and itinerary positions that correspond to this segment.
    '''
    def __init__(self, circuits, priority_circuits):
        self.train_id = idx_tp(-1)
        self.circuits = circuits
        self.priority_circuits = priority_circuits
        self.util_stats = List.empty_list(time_tuple_tp)

    def occupy(self, tr_id, acum_stats, current_tm):
        '''
        Occupy the segment by a given train ID. Store the current time as stating time for the utilization statistics, with still unknown duration (code -1).
        '''
        self.train_id = tr_id
        if acum_stats:
            if len(self.util_stats) > 0:
                last_end_tm = self.util_stats[-1][1]
                if last_end_tm > current_tm:
                    self.util_stats[-1] = (self.util_stats[-1][0], current_tm)
            self.util_stats.append((current_tm, time_tp(-1.0)))

    def free(self, acum_stats, free_tm):
        '''
        Mark the segment as free (-1). Fill the occupation duration for the statistics interval.
        '''
        self.train_id = idx_tp(-1)
        if acum_stats:
            start_tm = self.util_stats[-1][0]
            self.util_stats[-1] = (start_tm, free_tm - start_tm)

    def add_util(self, extra_tm):
        '''
        Add extra time to the last utilization interval. This is called when there is a train meet and the entering train is forced to wait.
        '''
        current_tup = self.util_stats[-1]
        self.util_stats[-1] = (current_tup[0], current_tup[1] + extra_tm)

# Circuit class attributes
spec_circuit = [
    ('train_ids', idx_arr_tp), # List of train IDs from this circuit
    ('len', idx_tp), # Length of the itinerary (number of directed segments per cycle)
    ('cycle_tm', time_tp), # Duration of a full ideal cycle (movement and scheduled processes)
    ('acum_exit_tm', time_array_tp), # Cumulative time for destination station exit, starting from the first segment, for a double cycle on closed circuits
    ('acum_entry_tm', time_array_tp), # Cumulative time for destination station entry, starting from the first segment
    ('segment_ids', idx_arr_tp), # List of segment IDs (non directed) from this circuit
    ('dest_st_id', idx_arr_tp), # IDs of destination stations per segment
    ('waiter_switch_tm', time_array_tp), # Time it takes for the waiter train to operate switches at the destination station when doing a meet
    ('passer_switch_tm', time_array_tp), # Time it takes for the priority train to operate switches at the destination station when doing a meet
    ('cooldown_tm', time_array_tp), # Time it takes for the station to process the next train from any circuit after the current train leaves
    ('origin_st_id', idx_tp), # ID of origin station for the first segment
    ('next_spawn', idx_tp), # Train ID for the next spawn, for open circuits
    ('frequency', time_tp), # Ideal time between trains in minutes
]

@jitclass(spec_circuit)
class Circuit:
    '''
    The Circuit has no class methods, but stores mainly static information about the circuit, such as the train IDs, the travel and process times,
    the segment IDs by traversal order, the stations IDs, the waiting times in case of a meet.
    The next_spawn attributes is dynamic, and iterates over the train IDs when spawning.
    '''
    def __init__(self, acum_exit_tm, acum_entry_tm, segment_ids, dest_st_id, 
                 waiter_switch_tm, passer_switch_tm, cooldown_tm, origin_st_id, next_spawn, frequency
                ):
        self.acum_entry_tm = acum_entry_tm
        self.len = idx_tp(len(self.acum_entry_tm))
        self.acum_exit_tm = acum_exit_tm
        self.cycle_tm = self.acum_exit_tm[self.len - 1]
        self.segment_ids = segment_ids
        self.dest_st_id = dest_st_id
        self.waiter_switch_tm = waiter_switch_tm
        self.passer_switch_tm = passer_switch_tm
        self.cooldown_tm = cooldown_tm
        self.origin_st_id = idx_tp(origin_st_id)
        self.next_spawn = idx_tp(next_spawn)
        self.frequency = frequency
        