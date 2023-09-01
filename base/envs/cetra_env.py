import gym
from gym import spaces
import pandas as pd
import numpy as np


class CetraEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 detectors,  # Action labels
                 cost_func,  # Cost function
                 t_func,  # Reward for TP or TN
                 fp_func,  # Reward for FP
                 fn_func,  # Reward for FN
                 illegal_reward  # Cost for illegal action
                 ):

        # init action space
        self.detectors = detectors
        self._termination_actions = ['benign', 'phishing']
        self._scan_actions = list(self.detectors.keys())
        self.action_labels = self._scan_actions + self._termination_actions
        self.action_space = spaces.Discrete(n=len(self.action_labels))

        # init state
        self.observation_space = spaces.Box(low=-1, high=1, shape=(len(self._scan_actions), 1))
        self._term_state = np.full(shape=(self.observation_space.shape[0],), fill_value=-1, dtype=np.float32)

        # init cost_func
        self.cost_func = cost_func

        # init rewards
        self.illegal_reward = illegal_reward
        self.t_func = t_func
        self.fp_func = fp_func
        self.fn_func = fn_func

    def step(self, action):  # action = discrete value : 0 - len(action_labels)-1
        action_label = self.action_labels[action]
        self.scanned.append(action_label)

        if action_label in self._scan_actions:
            if self.scanned.count(action_label) > 1:
                self.done = True
                # print("duplicate action!")
                return self.step_illegal_(pred='DUP')  # Duplicate action
            return self.step_scan_(action)

        elif action_label in self._termination_actions:
            self.done = True
            if len(self.scanned) == 1:
                # print("direct action!")
                return self.step_illegal_(pred='DIR')  # Direct classification
            return self.step_term_(action_label)

        raise ValueError('action value is outside of action_space')

    def apply_detectors(self, action, sample):
        action_label = self.action_labels[action]
        result, cost = self.detectors[action_label].detect(sample)  # detection place!
        return result, cost

    def step_scan_(self, action):
        # print(self.current['URL'])
        result, cost = self.apply_detectors(action, self.current)  # detection place!
        self.costs = np.append(self.costs, [cost], axis=0)
        self.state[action] = result  # update state
        self.reward += self.calc_func_cost(cost)  # update reward
        return self.state, 0, self.done, {}

    def step_illegal_(self, pred: str):
        self.state = self._term_state
        self.reward = self.illegal_reward
        self.pred = pred
        self.costs = np.array([self.illegal_reward])
        return self.state, self.reward, self.done, {'pred': pred}

    def step_term_(self, action_label):
        if self.current['label'] == 1:
            # Actual phishing
            if action_label == 'phishing':
                self.reward = self.t_func(self.reward)
                self.pred = 'TP'
            else:
                self.reward = self.fn_func(self.reward)
                self.pred = 'FN'
        else:
            # Actual benign
            if action_label == 'benign':
                self.reward = self.t_func(self.reward)
                self.pred = 'TN'
            else:
                self.reward = self.fp_func(self.reward)
                self.pred = 'FP'

        return self.state, self.reward, self.done, {}

    def _reset(self, url):
        self.current = url
        self.state = np.full(shape=(self.observation_space.shape[0],), fill_value=-1, dtype=np.float32)
        self.scanned = []
        self.reward = 0
        self.done = False
        self.pred = ''
        self.costs = np.array([], dtype=np.float32)

    def reset(self, url):
        self._reset(url)
        return self.state

    def calc_func_cost(self, cost):
        return self.cost_func(cost)

    def get_pred_and_costs(self):
        return self.pred, self.costs

    def get_benign_phish_labels_indices(self):
        last_idx = len(self.action_labels) - 1
        phish_idx = last_idx
        benign_idx = last_idx - 1
        return benign_idx, phish_idx

    def set_pertubed_state(self, pertubed_state):
        self.state = np.copy(pertubed_state)

    def render(self, mode='human', close=False):
        if self.done:
            label = 'm' if self.current['label'] == 1 else 'b'
            name = self.current['url']
            actions = ','.join(self.scanned)
            print(f'[{label}-{name}] Pred={self.pred:<3} R={self.reward:>8} A={actions}')
