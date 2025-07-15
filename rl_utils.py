import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen = capacity)

    def add(self, band_actor_inputs, width_actor_inputs, states, actions, reward, next_states, done, band_width_pluses, highs, lows):
        self.buffer.append((band_actor_inputs, width_actor_inputs, states, actions, reward, next_states, done, band_width_pluses, highs, lows))

#########################################################################
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        band_actor_inputs, width_actor_inputs, states, actions, reward, next_states, done, band_width_pluses, highs, lows = zip(*transitions)
        return [band_actor_inputs, width_actor_inputs, states, actions, reward, next_states, done, band_width_pluses, highs, lows]
        # band_actor_inputs = np.array(transitions[0][0][0])
        # width_actor_inputs = np.array(transitions[0][1][0])
        # states = np.array(transitions[0][2][0])
        # actions = np.array(transitions[0][3][0])
        # reward = transitions[0][4][0]
        # done = next_states = np.array(transitions[0][5][0])
        # band_width_pluses = transitions[0][6][0]
        # band_width_pluses = np.array(transitions[0][7][0])
        # print(len(transitions[0][0]))
        # # return np.array(transitions[0][0]), np.array(transitions[0][1]), np.array(transitions[0][2]), np.array(transitions[0][3]), transitions[0][4], np.array(transitions[0][5]), transitions[0][6], np.array(transitions[0][7])
        # return np.array(band_actor_inputs), np.array(width_actor_inputs), np.array(states), actions, reward, np.array(next_states), done, np.array(band_width_pluses), highs, lows

    def size(self):
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))