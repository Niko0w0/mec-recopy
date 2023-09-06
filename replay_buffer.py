import numpy as np
import random
import torch

class ReplayBuffer(object):
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    
    def __init__(self, size, alpha=0.6):
        self.storage = []
        self.priorities = np.zeros(size, dtype=np.float32)
        self.maxsize = int(size)
        self.next_idx = 0
        self.alpha = alpha
        self.epsilon = 1e-6

    def __len__(self):
        return len(self.storage)

    def clear(self):
        self.storage = []
        self.priorities = np.zeros(self.maxsize, dtype=np.float32)
        self.next_idx = 0

    def add(self, o, a, r, o_):
        data = (o, a, r, o_)
        priority = max(self.priorities.max(), self.epsilon)
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        self.priorities[self.next_idx] = priority
        self.next_idx = (self.next_idx + 1) % self.maxsize
     

    def sample(self, batch_size, agent_idx, beta=0.4):
        if len(self.storage) == 0:
            return None, None, None, None, None
        
        total = len(self.storage)
        
        priorities = self.priorities[:total]
        
        probabilities = priorities ** self.alpha
        
        # 如果出现NAN数据
        random_values = np.zeros(len(probabilities))
        nan_indices = np.isnan(probabilities)
        probabilities[nan_indices] = random_values[nan_indices]
        
        probabilities /= probabilities.sum()
        
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        idxes = np.random.choice(total, batch_size, replace=False, p=probabilities)
        obs, actions, rewards, obs_ = self.encode_sample(idxes, agent_idx)

        # Compute importance sampling weights
        weights = (total * probabilities[idxes]) ** (- beta)
        weights /= weights.max()

        return obs, actions, rewards, obs_, weights, idxes



    def encode_sample(self, idxes, agent_idx):
        observations, actions, rewards, observations_ = [], [], [], []
        
        for i in idxes:
            data = self.storage[i]
            obs, act, rew, obs_ = data
            observations.append(np.concatenate(obs[:]))
            actions.append(act)
            rewards.append(rew)
            observations_.append(np.concatenate(obs_[:]))
            
        return np.array(observations), np.array(actions), np.array(rewards), np.array(observations_)

    def make_index(self, batch_size):
        return [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]

    def update_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon
    
    def update_priorities(self, idxes, priorities):
        priorities += self.epsilon
        for idx, priority in zip(idxes, priorities):
            self.priorities[idx] = priority

