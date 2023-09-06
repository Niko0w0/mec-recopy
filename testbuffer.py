import numpy as np
import random

from replay_buffer import ReplayBuffer

buffter = ReplayBuffer(100)
o = []
obs = np.array([238,  212,  228,  202,  213,  168,  180,  171,  195, 191,  311,  306,  408,  384,  351])
a = np.array([238,  212,  228,  202,  213,  175,  159,  165,  189, 189,  381,  416,  443,  488,  317])
o.append(obs)
o.append(a)
a = []
action1 = np.array([1, 0.1])
action2 = np.array([2, 0.2])
a.append(action1)
a.append(action2)
r = [1000, 2000]
o_ = []
obs_1 = np.array([38,  212,  228,  202,  213,  175,  159,  165,  189, 189,  381,  416,  443,  488,  317])
obs_2 = np.array([38,  212,  228,  202,  213,  175,  159,  165,  189, 189,  381,  416,  443,  488,  317])
o_.append(obs_1)
o_.append(obs_2)
buffter.add(o, a, r, o_)

obs, action, reward, obs_ = buffter.sample(1, 1)

print(obs)
print(action)
print(reward)
print(obs_)