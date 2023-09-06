import copy
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 导入模块
sns.set() # 设置美化参数，一般默认就好

from env import ENV
from replay_buffer import ReplayBuffer
from MADDPG import Maddpg
from DQN import Double_DQN
from D3QN import D3QN
# from DDPG import DDPG

learning_start_step = 200
learning_fre = 5
batch_size = 64
gamma = 0.9
lr = 0.01
max_grad_norm = 0.5
save_model = 40
save_dir = "models/simple_adversary"
save_fer = 400
tao = 0.01
memory_size = 2000
EPOCH = 100
STEP = 200

write = SummaryWriter(log_dir="logs")
seed = 77378925
np.random.seed(seed)

def train(ue = 3, mec = 7, k = 11 * 3, lam = 0.5):
    """step1:create the environment"""
    u = ue
    m = mec
    k = k
    lam = lam
    env = ENV(u, m, k, lam)
    maddpg = Maddpg()
    dqn = Double_DQN(env)
    d3qn = D3QN(env)
    # ddpg = DDPG(env)


    print('=============================')
    print('=1 Env {} is right ...')
    print('=============================')

    """step2:create agent"""
    obs_shape_n = [env.n_features for i in range(env.UEs)]
    action_shape_n = [env.n_actions for i in range(env.UEs)]
    actors_cur, critic_cur, actors_tar, critic_tar, optimizers_a, optimizers_c = \
        maddpg.get_train(env, obs_shape_n, action_shape_n)
    memory_dpg = ReplayBuffer(memory_size)
    
    print('=============================')
    print(obs_shape_n)
    print(action_shape_n)
    # ddpg_actors_cur, ddpg_critic_cur, ddpg_actors_tar, ddpg_critic_tar, ddpg_optimizers_a, ddpg_optimizers_c = \
    #     ddpg.get_train(env, obs_shape_n[0], action_shape_n[0])

    print('=2 The {} agents are inited ...'.format(env.UEs))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    game_step = 0
    update_cnt = 0
    ddpg_update_cnt = 0
    episode_rewards, episode_dqn, episode_d3qn, episode_local,  episode_mec, episode_ran, episode_ddpg = [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0] # sum of rewards for all agents
    episode_time_dpg,  episode_time_dqn, episode_time_d3qn, episode_time_local, episode_time_ran, episode_time_mec, episode_time_ddpg = [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
    episode_energy_dpg, episode_energy_dqn, episode_energy_d3qn, episode_energy_local, episode_energy_ran, episode_energy_mec, episode_energy_ddpg = [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
    episode_total_cost = [0.0]
    epoch_average_reward, epoch_average_dqn, epoch_average_d3qn, epoch_average_local, epoch_average_mec, epoch_average_ran, epoch_average_ddpg= [], [], [], [], [], [], []
    epoch_average_time_reward, epoch_average_time_dqn, epoch_average_time_d3qn, epoch_average_time_local, epoch_average_time_mec, epoch_average_time_ran, epoch_average_ddpg= [], [], [], [], [], [], []
    epoch_average_energy_reward, epoch_average_energy_dqn, epoch_average_energy_d3qn, epoch_average_energy_local, epoch_average_energy_mec, epoch_average_energy_ran, epoch_average_energy_ddpg = [], [], [], [], [], [], []
    epoch_average_total_cost = []

    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a

    print('=3 starting iterations ...')
    print('=============================')

    for epoch in range(EPOCH):
        obs = env.reset()

        for time_1 in range(STEP):

            action_prob = [agent(torch.from_numpy(observation).to(torch.float)).detach().cpu().numpy() \
                        for agent, observation in zip(actors_cur, obs)]
            action_dqn = dqn.choose_action(obs)
            action_d3qn = d3qn.choose_action(obs)

            o1 = copy.deepcopy(obs)
            o2 = copy.deepcopy(obs)
            obs_old = copy.deepcopy(obs)
            obs_, rew, local, mec, ran, time_dpg, time_local, time_mec, time_ran, energy_dpg, energy_local, energy_mec, energy_ran, total_cost = env.step(obs, action_prob)
            obs_dqn, rew_dqn, time_dqn, energy_dqn = env.step(o1, action_dqn, is_prob=False, is_compared=False)
            obs_d3qn, rew_d3qn, time_d3qn, energy_d3qn = env.step(o2, action_d3qn, is_prob=False, is_compared=False)


            # save the expeeinece
            memory_dpg.add(obs_old, np.concatenate(action_prob), rew, obs_)
            dqn.store_memory(obs_old, action_dqn, rew_dqn, obs_dqn)
            d3qn.store_memory(obs_old, action_d3qn, rew_d3qn, obs_d3qn)
            
            # 这段代码是在训练过程中累积各种奖励或代价，并将其添加到相应的记录中。每个智能体的奖励和代价都会在每一轮训练中累加，并在训练的不同阶段进行统计和分析。
            episode_rewards[-1] += np.sum(rew)
            # 将当前时间步的奖励（rew）累加到最后一个轮次（episode_rewards）的奖励中。
            episode_dqn[-1] += np.sum(rew_dqn)
            episode_d3qn[-1] += np.sum(rew_d3qn)
            episode_local[-1] += np.sum(local)
            episode_mec[-1] += np.sum(mec)
            episode_ran[-1] += np.sum(ran)

            episode_time_dpg[-1] += np.sum(time_dpg)
            episode_time_dqn[-1] += np.sum(time_dqn)
            episode_time_d3qn[-1] += np.sum(time_d3qn)
            episode_time_local[-1] += np.sum(time_local)
            episode_time_ran[-1] += np.sum(time_ran)
            episode_time_mec[-1] += np.sum(time_mec)

            episode_energy_dpg[-1] += np.sum(energy_dpg)
            episode_energy_dqn[-1] += np.sum(energy_dqn)
            episode_energy_d3qn[-1] += np.sum(energy_d3qn)
            episode_energy_local[-1] += np.sum(energy_local)
            episode_energy_mec[-1] += np.sum(energy_mec)
            episode_energy_ran[-1] += np.sum(energy_ran)
            episode_total_cost[-1] += np.sum(total_cost)
            # for i, rew in enumerate(rew):agent_rewards[i][-1] += rew

            # train agent
            if game_step > 1000 and game_step % 100 == 0:
                update_cnt, actors_cur, actors_tar, critic_cur, critic_tar = maddpg.agents_train(
                    game_step, update_cnt, memory_dpg, obs_size, action_size,
                    actors_cur, actors_tar, critic_cur, critic_tar, optimizers_a, optimizers_c, write)
                
                # ddpg_update_cnt, ddpg_actors_cur, ddpg_actors_tar, ddpg_critic_cur, ddpg_critic_tar = ddpg.agents_train(
                #     game_step, ddpg_update_cnt, memory_dpg, obs_size, action_size,
                #     ddpg_actors_cur, ddpg_actors_tar, ddpg_critic_cur, ddpg_critic_tar, ddpg_optimizers_a, ddpg_optimizers_c, write)
                dqn.learn(game_step, write)
                d3qn.learn(game_step, write)
            
            
            
            # update obs
            game_step += 1
            obs = obs_
        epoch_average_reward.append(- episode_rewards[-1] / (env.UEs * STEP))
        epoch_average_dqn.append(- episode_dqn[-1] / (env.UEs * STEP))
        epoch_average_d3qn.append(- episode_d3qn[-1] / (env.UEs * STEP))
        epoch_average_local.append(episode_local[-1] / (env.UEs * STEP))
        epoch_average_mec.append(episode_mec[-1] / (env.UEs * STEP))
        epoch_average_ran.append(episode_ran[-1] / (env.UEs * STEP))

        epoch_average_time_reward.append(episode_time_dpg[-1] / (env.UEs * STEP))
        epoch_average_time_dqn.append(episode_time_dqn[-1] / (env.UEs * STEP))
        epoch_average_time_d3qn.append(episode_time_d3qn[-1] / (env.UEs * STEP))
        epoch_average_time_local.append(episode_time_local[-1] / (env.UEs * STEP))
        epoch_average_time_mec.append(episode_time_mec[-1] / (env.UEs * STEP))
        epoch_average_time_ran.append(episode_time_ran[-1] / (env.UEs * STEP))

        epoch_average_energy_reward.append(episode_energy_dpg[-1] / (env.UEs * STEP))
        epoch_average_energy_dqn.append(episode_energy_dqn[-1] / (env.UEs * STEP))
        epoch_average_energy_d3qn.append(episode_energy_d3qn[-1] / (env.UEs * STEP))
        epoch_average_energy_local.append(episode_energy_local[-1] / (env.UEs * STEP))
        epoch_average_energy_mec.append(episode_energy_mec[-1] / (env.UEs * STEP))
        epoch_average_energy_ran.append(episode_energy_ran[-1] / (env.UEs * STEP))
        epoch_average_total_cost.append(episode_total_cost[-1] / (env.UEs * STEP))

        episode_rewards.append(0)
        episode_dqn.append(0)
        episode_d3qn.append(0)
        episode_local.append(0)
        episode_mec.append(0)
        episode_ran.append(0)

        episode_time_dpg.append(0)
        episode_time_dqn.append(0)
        episode_time_d3qn.append(0)
        episode_time_local.append(0)
        episode_time_mec.append(0)
        episode_time_ran.append(0)

        episode_energy_dpg.append(0)
        episode_energy_dqn.append(0)
        episode_energy_d3qn.append(0)
        episode_energy_local.append(0)
        episode_energy_mec.append(0)
        episode_energy_ran.append(0)

        episode_total_cost.append(0)
        # for a_r in agent_rewards:
        #     a_r.append(0)
        # print("------reset-------")
        write.add_scalars("cost", {'MADDPG': epoch_average_total_cost[epoch],
                                   'DQN': epoch_average_dqn[epoch],
                                   'D3QN': epoch_average_d3qn[epoch],
                                   'Local': epoch_average_local[epoch],
                                   'Mec': epoch_average_mec[epoch],
                                   'random': epoch_average_ran[epoch]}, epoch)
        write.add_scalars("cost/energy", {'MADDPG': epoch_average_energy_reward[epoch],
                                     'DQN': epoch_average_energy_dqn[epoch],
                                     'D3QN': epoch_average_energy_d3qn[epoch],
                                     'Local': epoch_average_energy_local[epoch],
                                     'Mec': epoch_average_energy_mec[epoch],
                                     'random': epoch_average_energy_ran[epoch]}, epoch)
        write.add_scalars("cost/delay", {'MADDPG': epoch_average_time_reward[epoch],
                                    'DQN': epoch_average_time_dqn[epoch],
                                    'D3QN': epoch_average_time_d3qn[epoch],
                                    'Local': epoch_average_time_local[epoch],
                                    'Mec': epoch_average_time_mec[epoch],
                                    'random': epoch_average_time_ran[epoch]}, epoch)
        print("epoch:{},MADDPG:{}".format(epoch, epoch_average_total_cost[epoch]))
        print("epoch:{},DQN:{}".format(epoch, epoch_average_dqn[epoch]))
        print("epoch:{},Local:{}".format(epoch, epoch_average_local[epoch]))
        print("epoch:{},Mec:{}".format(epoch, epoch_average_mec[epoch]))
        print("epoch:{},random:{}".format(epoch, epoch_average_ran[epoch]))
        if epoch_average_mec[epoch] > epoch_average_reward[epoch]:
            print("True")
        print("---------------------------------------")
    
    
    average_epoch_total_cost = []
    average_epoch_dqn_cost = []
    average_epoch_d3qn_cost = []
    average_epoch_local_cost = []
    average_epoch_mec_cost = []
    average_epoch_ran_cost = []

    # 每一百个取一次平均，做平滑处理
    for i in range(0, len(epoch_average_total_cost), 100):
        avg_cost_maddpg = sum(epoch_average_total_cost[i : i + 100]) / 100
        average_epoch_total_cost.append(avg_cost_maddpg)
    for i in range(0, len(epoch_average_dqn), 100):
        avg_cost_dqn = sum(epoch_average_dqn[i : i + 100]) / 100
        average_epoch_dqn_cost.append(avg_cost_dqn)
    for i in range(0, len(epoch_average_d3qn), 100):
        avg_cost_d3qn = sum(epoch_average_d3qn[i : i + 100]) / 100
        average_epoch_d3qn_cost.append(avg_cost_d3qn)
    for i in range(0, len(epoch_average_local), 100):
        avg_cost_local = sum(epoch_average_local[i : i + 100]) / 100
        average_epoch_local_cost.append(avg_cost_local)
    for i in range(0, len(epoch_average_mec), 100):
        avg_cost_mec = sum(epoch_average_mec[i : i + 100]) / 100
        average_epoch_mec_cost.append(avg_cost_mec)
    for i in range(0, len(epoch_average_ran), 100):
        avg_cost_ran = sum(epoch_average_ran[i : i + 100]) / 100
        average_epoch_ran_cost.append(avg_cost_ran)
    
    # 画个图
    plt.plot(np.arange(len(average_epoch_total_cost)), average_epoch_total_cost, 'OliveDrab')
    plt.plot(np.arange(len(average_epoch_dqn_cost)), average_epoch_dqn_cost, 'Salmon')
    plt.plot(np.arange(len(average_epoch_d3qn_cost)), average_epoch_d3qn_cost, 'Tan')
    plt.plot(np.arange(len(average_epoch_local_cost)), average_epoch_local_cost, 'MediumPurple')
    plt.plot(np.arange(len(average_epoch_mec_cost)), average_epoch_mec_cost, 'OrangeRed')
    plt.plot(np.arange(len(average_epoch_ran_cost)), average_epoch_ran_cost, 'LightGreen')
    plt.legend(("MADDPG", "DQN", "D3QN", "Local", "Mec", "random"))
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()
    
    average_epoch_reward = []

    # 每一百个取一次平均，做平滑处理
    for i in range(0, len(epoch_average_reward), 100):
        avg_reward = sum(epoch_average_reward[i : i + 100]) / 100
        average_epoch_reward.append(avg_reward)
    
    plt.plot(np.arange(len(average_epoch_reward)), average_epoch_reward, 'Salmon')
    plt.legend(("reward"))
    plt.ylabel('reward')
    plt.xlabel('training steps')
    plt.show()
    
    epoch_average_total_cost = np.array(epoch_average_total_cost, dtype=float)
    epoch_average_dqn = np.array(epoch_average_dqn, dtype=float)
    epoch_average_d3qn = np.array(epoch_average_d3qn, dtype=float)
    epoch_average_local = np.array(epoch_average_local, dtype=float)
    epoch_average_mec = np.array(epoch_average_mec, dtype=float)
    epoch_average_ran = np.array(epoch_average_ran, dtype=float)
    epoch_average_reward = np.array(epoch_average_reward, dtype=float)
    
    # "MADDPG", "DQN", "D3QN", "Local", "Mec", "random"
    data = []
    data.append(epoch_average_total_cost)
    data.append(epoch_average_dqn)
    data.append(epoch_average_d3qn)
    data.append(epoch_average_local)
    data.append(epoch_average_mec)
    data.append(epoch_average_ran)
    data.append(epoch_average_reward)
    
    data = smooth(data, 80) # 平滑处理
    df = pd.DataFrame(data)
    df.to_csv('res/cost_total.csv', index=False)  # 将数据保存到名为 'your_data.csv' 的文件中
    
    # 设置 Seaborn 样式
    sns.set_style("whitegrid")

    # 创建一个图表
    plt.figure(figsize=(20, 12))  # 设置图表大小

    # 使用 Seaborn 画折线图
    sns.lineplot(data=df, x='Training Steps', y='MADDPG Cost', label='MADDPG')
    sns.lineplot(data=df, x='Training Steps', y='DQN Cost', label='DQN')
    sns.lineplot(data=df, x='Training Steps', y='D3QN Cost', label='D3QN')
    sns.lineplot(data=df, x='Training Steps', y='Local Cost', label='Local')
    sns.lineplot(data=df, x='Training Steps', y='Mec Cost', label='Mec')
    sns.lineplot(data=df, x='Training Steps', y='Random Cost', label='Random')

    # 添加标题和标签
    plt.title('Cost')
    plt.xlabel('Training Steps')
    plt.ylabel('Cost')

    # 添加图例
    plt.legend(title='Dataset', labels=['MADDPG', 'DQN', 'D3QN', 'Local', 'Mec', 'Random'])

    # 显示图表
    plt.show()
    
    # 创建一个图表
    plt.figure(figsize=(20, 12))  # 设置图表大小

    # 使用 Seaborn 画折线图
    sns.lineplot(data=df, x='Training Steps', y='reward', label='reward')

    # 添加标题和标签
    plt.title('Reward')
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')

    # 添加图例
    plt.legend(title='Dataset', labels='Reward')

    # 显示图表
    plt.show()
    
    # end of game
    print('game over')
    
def smooth(data, sm = 1):
    smooth_data = []
    if sm > 1:
        for d in data:
            z = np.ones(len(d))
            y = np.ones(sm) * 1.0
            d = np.convolve(y, d, "same") / np.convolve(y, z, "same")
            smooth_data.append(d)
    return smooth_data

if __name__ == '__main__':
    train()
