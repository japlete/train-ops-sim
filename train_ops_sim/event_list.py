import numpy as np
from numba import types, uint16, float64, int8, bool_, int64
from numba.experimental import jitclass

time_tp = float64
event_tp = int8
idx_tp = uint16

spec = [
    ('_times', types.Array(time_tp,1,'C')), # Simulation times in minutes
    ('_event_ids', types.Array(event_tp,1,'C')), # Action codes to be executed
    ('size', idx_tp), # Number of events in list
    ('_arr_size', idx_tp), # Underlying array size
    ('_has_peeked', bool_), # Whether the top time has been extracted through peek
]

@jitclass(spec)
class EventList:
    '''
    Handles the simulation event list. Stores the event occurrence times and action code, and provides methods to push and pop events.
    The event list is implemented as a binary heap, with the next event at index 0 (highest priority).
    '''
    def __init__(self):
        self._times = np.empty(16, dtype=time_tp)
        self._event_ids = np.empty(16, dtype=event_tp)
        self.size = idx_tp(0)
        self._arr_size = idx_tp(16)
        self._has_peeked = False

    def _parent(self, i : idx_tp):
        return (i - 1) // 2

    def _left_child(self, i : idx_tp):
        return 2 * i + 1

    def _right_child(self, i : idx_tp):
        return 2 * i + 2

    def _swap(self, i : idx_tp, j : idx_tp):
        self._times[i], self._times[j] = self._times[j], self._times[i]
        self._event_ids[i], self._event_ids[j] = self._event_ids[j], self._event_ids[i]

    def _sift_up(self, i : idx_tp):
        while i > 0 and self._times[self._parent(i)] > self._times[i]:
            self._swap(i, self._parent(i))
            i = self._parent(i)

    def _sift_down(self, i : idx_tp):
        while self._left_child(i) < self.size:
            min_child = self._left_child(i)
            if self._right_child(i) < self.size and self._times[self._right_child(i)] < self._times[min_child]:
                min_child = self._right_child(i)
            if self._times[i] <= self._times[min_child]:
                break
            self._swap(i, min_child)
            i = min_child

    def push_top(self, time : time_tp, action : event_tp):
        '''
        Replaces the top event in the event list with the given time and action code. This is only done if the top value has been extracted through a call to peek.
        '''
        self._times[0] = time
        self._event_ids[0] = action
        if self.size == idx_tp(0):
            self.size = idx_tp(1)
        self._sift_down(idx_tp(0))
    
    def push(self, time : time_tp, action : event_tp):
        '''
        Pushes a new event to the event list. If the event list is full, the underlying array is doubled in size to reserve memory for future events.
        '''
        if self._has_peeked:
            self._has_peeked = False
            self.push_top(time, action)
        else:
            if self.size == self._arr_size:
                self._arr_size *= 2
                new_times = np.empty(int64(self._arr_size), dtype=time_tp)
                new_times[:self.size] = self._times
                self._times = new_times
                new_event_ids = np.empty(int64(self._arr_size), dtype=event_tp)
                new_event_ids[:self.size] = self._event_ids
                self._event_ids = new_event_ids
            self._times[self.size] = time
            self._event_ids[self.size] = action
            self.size += 1
            self._sift_up(self.size - 1)

    def peek(self):
        '''
        Extracts the top event from the event list, without removing it. If the event list is empty, an IndexError is raised.
        '''
        if self.size == idx_tp(0):
            raise IndexError("Event list is empty")
        if self._has_peeked: # Doing 2 peeks in a row means the first didn't follow with a push_top, so manual pop is necessary
            self._times[0] = self._times[self.size - 1]
            self._event_ids[0] = self._event_ids[self.size - 1]
            self.size -= 1
            self._sift_down(idx_tp(0))
        self._has_peeked = True
        time = self._times[0]
        action = self._event_ids[0]
        return time, action

    def pop(self):
        '''
        Extracts the top event from the event list and removes it from the event list.
        '''
        time, action = self.peek()
        self._times[0] = self._times[self.size - 1]
        self._event_ids[0] = self._event_ids[self.size - 1]
        self.size -= 1
        self._sift_down(idx_tp(0))
        return time, action

    def sleep(self, delta_minutes : time_tp):
        '''
        Advances the time of all events in the event list by the given number of minutes. This simulates a system-wide halting of all operations.
        '''
        self._times[:self.size] += delta_minutes
