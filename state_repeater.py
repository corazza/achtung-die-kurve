from telnetlib import IP
import gym
import numpy as np
import IPython

def reshape_frame(frame, shape):
    return frame.reshape((*shape, 1))

class StateRepeater(gym.Wrapper):
    def __init__(self, env, history_length=4):
        super().__init__(env)
        self.env = env
        self.history_length = history_length
        self.repeated_state = None
        self._input_shape = self.env.observation_space.shape

    def reset(self):
        state, info = self.env.reset()
        self.frame = reshape_frame(state, self._input_shape)
        self.other_frame = reshape_frame(info['other_state'], self._input_shape)
        self.repeated_state = np.repeat(self.frame, self.history_length, axis=1)
        self.other_repeated_state = np.repeat(self.other_frame, self.history_length, axis=1)
        return state, info

    def step(self, action, render_mode=None):
        new_frame, reward, terminal, info = self.env.step(action)
        raw_frame = new_frame.copy()
        other_raw_frame = info['other_state'].copy()
        self.frame = reshape_frame(raw_frame, self._input_shape)
        self.repeated_state = np.append(self.repeated_state[:, 1:], self.frame, axis=1)
        self.other_frame = reshape_frame(other_raw_frame, self._input_shape)
        self.other_repeated_state = np.append(self.other_repeated_state[:, 1:], self.other_frame, axis=1)
        return raw_frame, reward, terminal, info
