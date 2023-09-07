import torch
import torch.nn as nn
import torch.nn.functional as F

class abstract_agent(nn.Module):

    def __init__(self):
        super().__init__()

    def act(self, x):
        policy, value = self.forward(x)
        return policy, value

class critic(abstract_agent):

    def __init__(self, obs_shape, act_shape):
        super().__init__()
        self.LRelu = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(act_shape + obs_shape, 64)
        self.linear_c2 = nn.Linear(64, 64)
        self.linear_c = nn.Linear(64, 1)

    def reset_parameters(self):
        # nn.init.xavier_uniform(self.linear_c1.weight, gain=nn.init.calculate_gain('leak_relu')) # 均匀分布
        # nn.init.xavier_uniform(self.linear_c2.weight, gain=nn.init.calculate_gain('leak_relu')) # 均匀分布
        # nn.init.xavier_uniform(self.linear_c.weight, gain=nn.init.calculate_gain('leak_relu')) # 均匀分布
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, act_input):
        print('forward')
        print(self, obs_input.shape, act_input.shape)
        # 将obs_input的维度调整为[64, 693]
        obs_input = obs_input.view(64, 693)
        x_cat = self.LRelu(self.linear_c1(torch.cat([obs_input, act_input], dim=1)))
        x = self.LRelu(self.linear_c2(x_cat))
        x = self.linear_c(x)

        return x

class actor(abstract_agent):

    def __init__(self, num_input, action_size):
        super().__init__()
        self.tanh = nn.Tanh()
        self.LRelu = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_input, 64)
        self.linear_a2 = nn.Linear(64, 64)
        self.linear_a = nn.Linear(64, action_size)

    def reset_parameters(self):
        # nn.init.xavier_uniform(self.linear_c1.weight, gain=nn.init.calculate_gain('leak_relu')) # 均匀分布
        # nn.init.xavier_uniform(self.linear_c2.weight, gain=nn.init.calculate_gain('leak_relu')) # 均匀分布
        # nn.init.xavier_uniform(self.linear_c.weight, gain=nn.init.calculate_gain('leak_relu')) # 均匀分布
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x, model_original_out=False):
        # print("x")
        # print(x)
        x = self.LRelu(self.linear_a1(x))
        # print("a1-------")
        # print(x)
        x = self.LRelu(self.linear_a2(x))
        # print("a1-------")
        # print(x)
        model_out = self.linear_a(x)
        u = torch.rand_like(model_out)
        policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        if model_original_out:
            return model_out, policy
        return policy