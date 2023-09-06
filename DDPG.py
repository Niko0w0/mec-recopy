import torch
import torch.nn as nn
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import json

from Rl_net import actor, critic

learning_start_step = 200
learning_fre = 5
batch_size = 64
gamma = 0.9
lr = 0.01
max_grad_norm = 0.5
save_model = 40
save_dir = "models/simple_adversary/ddpg"
save_fer = 400
tao = 0.01
file_path = "res/ddpg_td_errors.json"

class DDPG(object):
    def __init__(self, env):
        self.env = env
    
    def get_train(self, env, obs_shape_n, action_shape_n):
        
        actors_cur = None
        critics_cur = None
        actors_target = None
        critics_target = None
        optimizer_a = None
        optimizer_c = None

        actors_cur = actor(obs_shape_n, action_shape_n)
        critics_cur = critic(obs_shape_n, action_shape_n)
        actors_target = actor(obs_shape_n, action_shape_n)
        critics_target = critic(obs_shape_n, action_shape_n)
        optimizer_a = torch.optim.Adam(actors_cur.parameters(), lr = lr)
        optimizer_c = torch.optim.Adam(critics_cur.parameters(), lr = lr)
        actors_tar = self.update_train(actors_cur, actors_target, 1.0)
        critics_tar = self.update_train(critics_cur, critics_target, 1.0)
        return actors_cur, critics_cur, actors_tar, critics_tar, optimizer_a, optimizer_c

    def update_train(self, agents_cur, agents_tar, tao):
        """
        用于更新target网络，
        这个方法不同于直接复制，但结果一样
        out:
        |agents_tar: the agents with new par updated towards agents_current
        agents_cur: 当前智能体的策略网络列表（或者价值网络列表）。
        agents_tar: 目标智能体的策略网络列表（或者价值网络列表）。
        """
        key_list = list(agents_cur.state_dict().keys())
        state_dict_t = agents_tar.state_dict()
        state_dict_c = agents_cur.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key] * tao + \
                                (1 - tao) * state_dict_t[key]
        agents_tar.load_state_dict(state_dict_t)
        return agents_tar

    def agents_train(self, game_step, update_cnt, memory, obs_size, action_size,
                     actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, write):
        """
        par:
        |input: the data for training
        |output: the data for next update
        """
        # 训练
        if (game_step > learning_start_step) and (game_step % learning_fre == 0):
            if update_cnt == 0: print('\r=start training...' + ''*100)
            update_cnt += 1

            # 随机抽样
            rew = []
            agent_idx = 1
            obs, action, reward, obs_, weights, idxes = memory.sample(batch_size, agent_idx, beta=0.4)
            # print(obs, action, reward, obs_, weights, idxes)
            
            for i in range(batch_size):
                r = reward[i]
                # print('****************')
                # print("r:", r)
                # print("type(r):", type(r))
                # print('****************')
                ar = sum(r)/len(r)
                rew.append(ar)

            # update critic
            # rew = torch.tensor(reward, dtype=torch.float)
            rew = torch.tensor(rew, dtype=torch.float)
            action_cur = torch.from_numpy(action).to(torch.float)
            obs_n = torch.from_numpy(obs).to(torch.float)
            obs_n_ = torch.from_numpy(obs_).to(torch.float)
            # actors_tar = [actor(num_input, action_size) for _ in range(num_actors)]
            # action_tar = torch.cat([a_t(obs_n_[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
            #                         for idx, a_t in enumerate(actors_tar)], dim=1)
            action_tar = actors_tar(obs_n_[:, obs_size[0][0]:obs_size[0][1]]).detach()
            
            q = critics_cur(obs_n, action_cur).reshape(-1)     # q
            q_ = critics_tar(obs_n_, action_tar).reshape(-1)   # q_
            tar_value = q_ * gamma + rew
            loss_c = torch.nn.MSELoss()(q, tar_value)
            optimizers_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critics_cur.parameters(), max_grad_norm)
            optimizers_c.step()

            # update Actor
            # There is no need to cal other agent's action
            model_out, policy_c_new = actors_cur(
                obs_n_[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True)
            # update the action of this agent
            action_cur[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new
            loss_pse = torch.mean(torch.pow(model_out, 2))
            loss_a = torch.mul(-1, torch.mean(critics_cur(obs_n, action_cur)))

            optimizers_a.zero_grad()
            loss_t = 1e-3 * loss_pse + loss_a
            loss_t.backward()
            nn.utils.clip_grad_norm_(actors_cur.parameters(), max_grad_norm)
            optimizers_a.step()
            
            td_errors = []
            td_errors = tar_value - q
            # priorities = np.abs(td_errors)
            priorities = np.abs(td_errors.detach().numpy())
            
            with open(file_path, "a") as file:
                json.dump(td_errors.tolist(), file)
                file.write("\n")  # 写入换行符，以便每次更新都在文件最末添加数据
            
            memory.update_priorities(idxes, priorities)
            
            write.add_scalar("Loss/ddpg/Actor", loss_t, game_step)
            write.add_scalar("Loss/ddpg/Critic", loss_c, game_step)
                
            # save model
            if update_cnt > save_model and update_cnt % save_fer == 0:
                time_now = time.strftime('%y%m_%d%H%M')
                print('=time:{} step:{}        save'.format(time_now, game_step))
                model_file_dir = os.path.join(save_dir, '{}_{}'.format(time_now, game_step))
                if not os.path.exists(model_file_dir):  # make the path
                    os.makedirs(model_file_dir)
                
                torch.save(actors_cur, os.path.join(model_file_dir, 'a_c.pt'))
                torch.save(actors_tar, os.path.join(model_file_dir, 'a_t.pt'))
                torch.save(critics_cur, os.path.join(model_file_dir, 'c_c.pt'))
                torch.save(critics_tar, os.path.join(model_file_dir, 'c_t.pt'))

            # update the tar par
            actors_tar = self.update_train(actors_cur, actors_tar, tao)
            critics_tar = self.update_train(critics_cur, critics_tar, tao)
            
            
        return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar
