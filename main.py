'''
特征提取  Resnet101、convlstm
波段数 21、42、84
'''
import argparse
import os
import random
import rl_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# import env

def get_args():
    parser = argparse.ArgumentParser(
        "maddpg波段选择")
    parser.add_argument("--lr", type=int, default=10)
    parser.add_argument("--widthSelect_input_dim", type=int, default=5)
    parser.add_argument("--selected_bands_num", type=int, default=5)
    parser.add_argument("--bandState_dim", type=int, default=42)###42/21
    parser.add_argument("--file_path", type=str, default='./feature_data')
    parser.add_argument("--num_episodes", type=int, default=5000, help="总轮次")
    parser.add_argument("--episode_length", type=int, default=10, help="每条序列的最大长度")
    parser.add_argument("--buffer_size", type=int, default=1000)  # 10000
    # parser.add_argument("--hidden_dim", type = int, default = 64)
    parser.add_argument("--actor_lr", type=float, default=1e-2)
    parser.add_argument("--critic_lr", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=1)  # 1024
    parser.add_argument("--update_interval", type=int, default=100)
    parser.add_argument("--minimal_size", type=int, default=50)  # 4000
    arg = parser.parse_args()
    return arg


def onehot_from_logits(logits, eps=0.01):
    '''生成最优动作的独热（one-hot）形式'''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作，转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]], requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Sotfmax分布中采样，并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的读热量，梯度为y，既能得到一个与环境交互的离散动作，又可以正确得反向传递梯度
    return y


# 波段选择actor
class BandSelect_Actor(nn.Module):
    def __init__(self, state_size, action_dim, linear_hidden_dim):
        super(BandSelect_Actor, self).__init__()
        self.state_size = state_size
        # 21  * 5 * 256 * 320
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(256, 320)),
            nn.ReLU()
        )
        # 21  * 10 * 1 * 1
        # 21  * 11
        self.fc1 = nn.Linear(11, 5)
        # 21  * 5
        self.fc2 = nn.Linear(5, 1)
        # 21  * 1
        # 21
        self.fc3 = nn.Linear(state_size, linear_hidden_dim)
        self.fc4 = nn.Linear(linear_hidden_dim, action_dim)
        # self.action_dim = action_dim
        # self.rnn = nn.GRU(rnn_input_dim, rnn_hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        # self.fc1 = nn.Linear(rnn_hidden_dim * 2, 1)
        # self.fc2 = nn.Linear(action_dim, linear_hidden_dim)
        # self.fc1 = nn.Sequential(
        #     nn.Linear(rnn_hidden_dim * 2, 1),
        #     nn.ReLU()
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(action_dim, linear_hidden_dim),
        #     nn.ReLU()
        # )
        # self.fc3 = nn.Linear(linear_hidden_dim, action_dim)

    def forward(self, x, band_width_plus):
        x = self.conv(x)
        x = x.squeeze()
        # plus_tensor = torch.zeros(21, 1).to(x.device)   ##
        plused_tensor = torch.cat((x, band_width_plus), dim=-1)
        x = F.relu(self.fc1(plused_tensor))
        x = F.relu(self.fc2(x))
        x = x.squeeze(-1)
        if x.ndim == 1:
            x = torch.unsqueeze(x, dim=0)
        x1 = F.relu(self.fc3(x))
        x1 = self.fc4(x1)
        x1 = torch.cat((x1[:, :self.state_size], torch.softmax(x1[:, self.state_size:], dim=-1)), dim=-1)
        # x, _ = self.rnn(x)
        # x = self.fc1(x)
        # x = x.squeeze(-1)
        # # x = torch.reshape(x, (1, self.action_dim))
        # x1 = self.fc2(x)
        # x1 = self.fc3(x1)

        return x, x1


# 波段选择critic
class BandSelect_Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(BandSelect_Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, critic_input):
        # x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(critic_input))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
    # def forward(self, state, action):
    #     x = torch.cat((state, action), dim=1)
    #     x = nn.ReLU(self.fc1(x))
    #     x = nn.ReLU(self.fc2(x))
    #     q_value = self.fc3(x)
    #     return q_value


# 带宽选择actor
class Bandwidth_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Bandwidth_Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state, high, low):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        one_tensor = torch.ones_like(x[:, 0:1])
        output_modified = torch.tanh(x[:, 0:1]) + one_tensor
        action = torch.cat(((output_modified / 2 * (high - low) + torch.tensor([[low]]).to(output_modified.device)),
                            torch.sigmoid(x[:, 1:])), dim=1)
        # tanh_output = torch.tanh(self.fc3(x))
        # # 缩放到[min, max]范围内
        # one_tensor = torch.ones_like(tanh_output)
        # output_modified = tanh_output + one_tensor
        # # 缩放到[low, high]范围内
        # action = output_modified / 2 * (high - low) + torch.tensor([[low]]).to(output_modified.device)
        # action = self.max_action * torch.tanh(x)
        # action_probs = self.softmax(x)
        return action


# 带宽选择critic
class Bandwidth_Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Bandwidth_Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, critic_input):
        # x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(critic_input))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
    # def forward(self, state, action):
    #     x = torch.cat((state, action), dim=1)
    #     x = nn.ReLU(self.fc1(x))
    #     x = nn.ReLU(self.fc2(x))
    #     q_value = self.fx3(x)
    #     return q_value


# 波段选择智能体
class BandSelect_Agent:
    def __init__(self, state_dim, action_dim, critic_input_dim, linear_hidden_dim, critic_hidden_dim,
                 actor_lr, critic_lr, device):
        self.state_dim = state_dim
        self.actor = BandSelect_Actor(state_dim, action_dim, linear_hidden_dim).to(device)
        self.target_actor = BandSelect_Actor(state_dim, action_dim, linear_hidden_dim).to(device)

        self.critic = BandSelect_Critic(critic_input_dim, critic_hidden_dim).to(device)
        self.target_critic = BandSelect_Critic(critic_input_dim, critic_hidden_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state, band_width_plus, explore=False):
        # with torch.no_grad():
        width_input, action = self.actor(state, band_width_plus)
        if explore:
            action = torch.cat((gumbel_softmax(action[:, :self.state_dim]), action[:, self.state_dim:]), dim=1)
        else:
            action = torch.cat((onehot_from_logits(action[:, :self.state_dim]), action[:, self.state_dim:]), dim=1)
        # action.detach().cpu().numpy()[0]
        return width_input, action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)


# 带宽选择智能体
class WidthSelect_Agent:
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim, critic_hidden_dim,
                 actor_lr, critic_lr, device):
        self.actor = Bandwidth_Actor(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = Bandwidth_Actor(state_dim, action_dim, hidden_dim, ).to(
            device)

        self.critic = Bandwidth_Critic(critic_input_dim, critic_hidden_dim).to(device)
        self.target_critic = Bandwidth_Critic(critic_input_dim, critic_hidden_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state, high, low):
        action = self.actor(state, high, low)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)


class make_Env:
    def __init__(self, bandState_dim, widthSelect_input_dim, selected_bands_num, filepath):
        # self.bandSelect_input_plus = bandSelect_input_plus
        self.widthSelect_input_dim = widthSelect_input_dim
        self.reward = 0
        self.train_index = 0
        self.filepath = filepath
        # self.data_num = int(len(os.listdir(filepath)))
        self.data_num = 18
        self.count = 0
        self.bands_num = selected_bands_num
        self.Entropy_total = np.load('allEntropy.npy')  # (18, 21)
        self.IG_total = np.load('allIG.npy')  # (18, 21)
        self.Detaction_total = np.load('detact.npy')  # (18,21,5,5)
        self.Lusudeng_total = np.load('penzui.npy')  # (18,21,5,5)
        self.Noise_total = np.load('noise.npy')  # (18,21,6,5)
        self.width_data = np.array([51.93, 49.47, 47.72, 46.09, 44.94, 44.2, 43.26, 42.51, 41.7,
                                    40.83, 39.89, 38.89, 37.86, 36.81, 36.1, 35.06, 34.03, 33.37, 32.41, 31.49,30.9])  # 每个波段的带宽
        self.bandState_dim = bandState_dim
        self.band_selected = torch.zeros(self.bandState_dim)
        self.band_width_selected = torch.zeros(self.bandState_dim)

    def reset(self):
        self.reward = 0
        self.count = 0
        self.data_index = random.randint(1, self.data_num)
        self.filename = self.filepath + '/feature' + str(self.data_index) + '.npy'

        # 21, 5, 256, 320
        self.multispec_feature = torch.from_numpy(np.load(self.filename))
        self.band_selected = torch.zeros(self.bandState_dim)
        self.band_width_selected = torch.zeros(self.bandState_dim, 1)
        self.max_entrpy_idx = np.argmax(self.Entropy_total[self.data_index - 1, :])
        # self.multispec_feature = torch.rand(1, 21, 512)
        # self.bandSelect_state_plus = torch.zeros(1, 21, 21)
        # self.bandSelect_state = torch.cat((self.multispec_feature, self.bandSelect_state_plus), 2)
        return [False, False], self.band_width_selected, self.multispec_feature

    def step(self, state, actions):
        '''选择动作，计算奖励'''
        self.count += 1
        # dones = True if self.count == self.bands_num else False
        if self.count == self.bands_num:
            dones = [True, True]
        else:
            dones = [False, False]
        '''归一化奖励系数'''
        a, b = actions[0][self.bandState_dim:].tolist()
        c = actions[1][1].item()
        # 归一化 a 和 b 的值
        ab_sum = a + b
        a = a / ab_sum * (1 - c)
        b = b / ab_sum * (1 - c)
        bandselect_action = np.argmax(actions[0][0:self.bandState_dim])
        '''band_select奖励'''
        # if(self.count == 1):
        #     if(bandselect_action != self.max_entrpy_idx):
        #         band_reward = -1
        #         self.band_selected[bandselect_action] = 1
        #         dones = [True, True]
        #     else:
        #         band_reward = 1
        #         self.band_selected[bandselect_action] = 1
        # else:
        if (self.band_selected[bandselect_action] == 1):
            band_reward = -1
            dones = [True, True]
        else:
            self.band_selected[bandselect_action] = 1
            '''r1计算'''
            r1 = self.IG_total[self.data_index - 1][bandselect_action]
            # r1 = self.Entropy_total[self.data_index - 1][bandselect_action]
            '''r2计算'''
            ones = self.band_selected == 1
            # 获取值为1的元素的索引
            indices = torch.nonzero(ones, as_tuple=False).cpu().numpy()
            vector_detact = self.Detaction_total[self.data_index - 1][indices]
            vector_detact_max = np.squeeze(np.max(vector_detact, axis=(2, 3)))
            vector_lusudeng = self.Lusudeng_total[self.data_index - 1][indices]
            vector_lusudeng_max = np.squeeze(np.max(vector_lusudeng, axis=(2, 3)))
            r2 = -cal_spectral_angle(vector_detact_max, vector_lusudeng_max)
            r1 *= a
            r2 *= b
            band_reward = r1 + r2

        '''计算r3  信杂加噪声比'''
        bandwidth = actions[1][0].item()
        '''波段合成'''
        if (bandwidth > self.width_data[bandselect_action] - 1 and bandwidth < self.width_data[bandselect_action] + 1):
            x = 1
            y = 0
            z = 0
        elif (bandwidth >= self.width_data[bandselect_action] + 1 and bandwidth < np.sum(self.width_data[bandselect_action - 1: bandselect_action + 2])):
            x = 1
            y = (bandwidth - self.width_data[bandselect_action]) / ( self.width_data[bandselect_action - 1] + self.width_data[bandselect_action + 1])
            z = 0
        else:
            x = 1
            y = 1
            z = (bandwidth - np.sum(self.width_data[bandselect_action - 1: bandselect_action + 2])) / (self.width_data[bandselect_action - 2] + self.width_data[bandselect_action + 2])
        x = 0.4 * x / (0.4 * x + 2 * 0.24 * y + 2 * 0.06 * z)
        y = 0.24 * y / (0.4 * x + 2 * 0.24 * y + 2 * 0.06 * z)
        z = 0.06 * z / (0.4 * x + 2 * 0.24 * y + 2 * 0.06 * z)
        if (bandselect_action == 0 or bandselect_action == 20):
            detaction = self.Detaction_total[self.data_index - 1, bandselect_action, :, :]
            lusudeng = self.Lusudeng_total[self.data_index - 1, bandselect_action, :, :]
            noise = self.Noise_total[self.data_index - 1, bandselect_action, :, :]
            self.band_width_selected[bandselect_action] += 1
        elif (bandselect_action == 1 or bandselect_action == 19):
            detaction = x * self.Detaction_total[self.data_index - 1, bandselect_action, :, :] + \
                        y * self.Detaction_total[self.data_index - 1, bandselect_action - 1, :,
                            :] + y * self.Detaction_total[self.data_index - 1, bandselect_action + 1, :, :]
            lusudeng = self.Lusudeng_total[self.data_index - 1, bandselect_action, :, :] + \
                       y * self.Lusudeng_total[self.data_index - 1, bandselect_action - 1, :,
                           :] + y * self.Lusudeng_total[self.data_index - 1, bandselect_action + 1, :, :]
            noise = self.Noise_total[self.data_index - 1, bandselect_action, :, :] + \
                    y * self.Noise_total[self.data_index - 1, bandselect_action - 1, :, :] + y * self.Noise_total[
                                                                                                 self.data_index - 1,
                                                                                                 bandselect_action + 1,
                                                                                                 :, :]
            self.band_width_selected[bandselect_action] += x
            self.band_width_selected[bandselect_action - 1] += y
            self.band_width_selected[bandselect_action + 1] += y
        else:
            detaction = self.Detaction_total[self.data_index - 1, bandselect_action, :, :] \
                        + y * self.Detaction_total[self.data_index - 1, bandselect_action - 1, :,
                              :] + y * self.Detaction_total[self.data_index - 1, bandselect_action + 1, :, :] \
                        + z * self.Detaction_total[self.data_index - 1, bandselect_action - 2, :,
                              :] + z * self.Detaction_total[self.data_index - 1, bandselect_action + 2, :, :]
            lusudeng = self.Lusudeng_total[self.data_index - 1, bandselect_action, :, :] \
                       + y * self.Lusudeng_total[self.data_index - 1, bandselect_action - 1, :,
                             :] + y * self.Lusudeng_total[self.data_index - 1, bandselect_action + 1, :, :] \
                       + z * self.Lusudeng_total[self.data_index - 1, bandselect_action - 2, :,
                             :] + z * self.Lusudeng_total[self.data_index - 1, bandselect_action + 2, :, :]
            noise = self.Noise_total[self.data_index - 1, bandselect_action, :, :] \
                    + y * self.Noise_total[self.data_index - 1, bandselect_action - 1, :, :] + y * self.Noise_total[
                                                                                                   self.data_index - 1,
                                                                                                   bandselect_action + 1,
                                                                                                   :, :] \
                    + z * self.Noise_total[self.data_index - 1, bandselect_action - 2, :, :] + z * self.Noise_total[
                                                                                                   self.data_index - 1,
                                                                                                   bandselect_action + 2,
                                                                                                   :, :]
            self.band_width_selected[bandselect_action] += x
            self.band_width_selected[bandselect_action - 1] += y
            self.band_width_selected[bandselect_action + 1] += y
            self.band_width_selected[bandselect_action - 2] += z
            self.band_width_selected[bandselect_action + 2] += z
        if torch.any(self.band_width_selected > 1.2):
            r3 = -1
            dones = [True, True]
        else:
            r3 = cal_scrsnr(detaction, lusudeng, noise) / 2
            r3 *= c

        # print(f'train_idx is {self.train_index} and bandselect-action is {bandselect_action}.')
        # IES = self.IES_total[self.train_index][bandselect_action]
        # band_reward = IES
        # width_reward = random.random()

        #########################################################################
        ########## 修改处
        ###########
        reward = [band_reward, r3]
        # reward = [band_reward, 0]
        next_state = state
        return next_state, reward, dones, self.band_width_selected


'''计算信杂加噪声比'''


def cal_scrsnr(detaction, lusudeng, noise):
    scr = abs(detaction.mean() - lusudeng.mean()) / lusudeng.std()
    snr = abs(detaction.mean() - noise.mean()) / noise.std()
    return 1 / (1 / scr + 1 / snr)


'''计算光谱角'''


def cal_spectral_angle(vector_A, vector_B):
    dot_product = np.dot(vector_A, vector_B)

    # 计算向量 A 和向量 B 的模长
    magnitude_A = np.linalg.norm(vector_A)
    magnitude_B = np.linalg.norm(vector_B)

    # 计算光谱角
    spectral_angle = np.arccos(dot_product / (magnitude_A * magnitude_B))
    return spectral_angle


def entropy(probabilities):
    """计算熵"""
    return -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))


def information_gain(image_entropies, joint_entropy):
    """计算信息增益"""
    return np.sum(image_entropies) - joint_entropy


def IG(image1_prob, image2_prob, image3_prob):
    # 计算每张图像的熵
    image1_entropy = entropy(image1_prob)
    image2_entropy = entropy(image2_prob)
    image3_entropy = entropy(image3_prob)

    # 计算联合概率
    joint_prob = (image1_prob + image2_prob + image3_prob) / 3

    # 计算联合熵
    joint_entropy = entropy(joint_prob)

    # 计算信息增益
    info_gain = information_gain([image1_entropy, image2_entropy, image3_entropy], joint_entropy)
    return info_gain


class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, linear_hidden_dims,
                 critic_hidden_dim, state_dims, action_dims, critic_input_dim, gamma, gau):
        self.env = env
        self.device = device
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.linear_hidden_dims = linear_hidden_dims
        self.critic_hidden_dim = critic_hidden_dim
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.critic_input_dim = critic_input_dim
        self.critic_criterion = torch.nn.MSELoss()
        self.gamma = gamma
        self.gau = gau
        self.bandSelect_Agent = BandSelect_Agent(self.state_dims[0], self.action_dims[0], self.critic_input_dim,
                                                 self.linear_hidden_dims[0], self.critic_hidden_dim,
                                                 self.actor_lr, self.critic_lr, self.device)
        self.widthSelect_Agent = WidthSelect_Agent(self.state_dims[1], self.action_dims[1], self.critic_input_dim,
                                                   self.linear_hidden_dims[1],
                                                   critic_hidden_dim, self.actor_lr, self.critic_lr, self.device)
        self.agents = [self.bandSelect_Agent, self.widthSelect_Agent]

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    # def take_action(self, state, explore):
    #     width_input, band_action = self.agents[0].take_action(state)
    #     width_input = width_input[..., 0:4]

    # states = [
    #     torch.tensor([states[i]], dtype=torch.float, device=self.device)
    #     for i in range(len(self.agents))
    # ]
    # return [
    #     agent.take_action(state, explore)
    #     for agent, state in zip(self.agents, states)
    # ]

    def update(self, sample, i_agent):
        # actor_inputs, obs, act, rew, next_actor_inputs, next_obs, done, band_width_pluses = sample
        band_actor_inputs, width_actor_inputs, obs, act, rew, next_obs, done, band_width_pluses, highs, lows = sample
        # band_actor-inputs: 波段选择智能体输入[0]代表cur，[1]代表next
        # width_actor_inputs: 带宽选择智能体输入[0]代表cur，[1]代表next
        # obs: critic的输入，波段选择智能体为中间[1,21]状态
        # band_width_pluses：已选波段和带宽数据，在波段选择智能体中与[21, 10]cat
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()
        _, band_output = self.target_policies[0](band_actor_inputs[1], band_width_pluses[1])
        # bandSelect_target_act = onehot_from_logits(band_output)
        bandSelect_target_act = torch.cat(
            (onehot_from_logits(band_output[:, :self.state_dims[0]]), band_output[:, self.state_dims[0]:]), dim=1)
        widthSelect_target_act = self.target_policies[1](width_actor_inputs[1], highs[1][0], lows[1][0])
        all_target_act = [bandSelect_target_act, widthSelect_target_act]
        # all_target_act = [
        #     onehot_from_logits(pi(_next_obs))
        #     for pi, _next_obs in zip(self.target_policies, next_obs)
        # ]
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(
            -1, 1) + self.gamma * cur_agent.target_critic(
            target_critic_input) * (1 - done[i_agent].view(-1, 1))
        # target_critic_value = rew[i_agent].view(
        #     -1, 1) + self.gamma * cur_agent.target_critic(
        #         *next_obs, *all_target_act) * (1 - done[i_agent].view(-1, 1))
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())
        # critic_loss = torch.nn.MSELoss(critic_value, target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        if (i_agent == 0):
            _, cur_actor_out = cur_agent.actor(band_actor_inputs[0], band_width_pluses[0])  # band智能体
        else:
            cur_actor_out = cur_agent.actor(width_actor_inputs[0], highs[0][0], lows[0][0])  # width智能体
        # cur_act_vf_in = gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        actor_inputs = [band_actor_inputs[0], width_actor_inputs[0]]
        for i, (pi, _actor_inputs) in enumerate(zip(self.policies, actor_inputs)):
            if i == 0:
                if i == i_agent:
                    _, put = pi(_actor_inputs, band_width_pluses[0])
                    # torch.cat((onehot_from_logits(put[:, :self.state_dims[0]]), band_output[:, self.state_dims[0]:]),
                    #           dim=1)
                    all_actor_acs.append(torch.cat(
                        (gumbel_softmax(put[:, :self.state_dims[0]]), band_output[:, self.state_dims[0]:]), dim=1))
                    # all_actor_acs.append(gumbel_softmax(put))
                else:
                    _, put = pi(_actor_inputs, band_width_pluses[0])
                    all_actor_acs.append(torch.cat(
                        (onehot_from_logits(put[:, :self.state_dims[0]]), band_output[:, self.state_dims[0]:]), dim=1))
                    # all_actor_acs.append(onehot_from_logits(put))
            else:
                all_actor_acs.append(pi(_actor_inputs, highs[0][0], lows[0][0]))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        # actor_loss = -cur_agent.critic(*obs, *all_actor_acs).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.gau)
            agt.soft_update(agt.critic, agt.target_critic, self.gau)


def evaluate(maddpg, n_episode=10, episode_length=25):
    # 对学习的策略进行评估,此时不会进行探索
    env = make_Env(opt.bandState_dim, opt.widthSelect_input_dim, opt.selected_bands_num, opt.file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # returns = np.zeros(len(env.agents))
    returns = np.zeros(2)
    for _ in range(n_episode):
        print(n_episode,_,returns)
        done, band_width_plus, state = env.reset()  # band_width_plus: 已选波段和带宽状态
        state = state.to(device)
        band_width_plus = band_width_plus.to(device)
        for t_i in range(episode_length):
            if (done[0] == True):
                dones, band_width_plus, state = env.reset()  # band_width_plus: 已选波段和带宽状态
                state = state.to(device)
                band_width_plus = band_width_plus.to(device)
            # actions = maddpg.take_action(obs, explore=False)
            actions = []
            width_input, band_action = maddpg.bandSelect_Agent.take_action(state, band_width_plus, explore=False)
            actions.append(band_action)
            selected_band = np.argmax(band_action[:21])
            if (selected_band == 0 or selected_band == 20):
                width_state = torch.zeros(1, 5).to(device)
                width_state.view(-1)[2] = width_input[0][selected_band]
                high = env.width_data[selected_band]
            elif (selected_band == 1 or selected_band == 19):
                width_state = torch.zeros(1, 5).to(device)
                width_state.view(-1)[2] = width_input[0][selected_band]
                width_state.view(-1)[1] = width_input[0][selected_band - 1]
                width_state.view(-1)[3] = width_input[0][selected_band + 1]
                high = np.sum(env.width_data[selected_band - 1:selected_band + 2])
            else:
                width_state = width_input[..., (selected_band - 2):(selected_band + 3)]
                high = np.sum(env.width_data[selected_band - 2:selected_band + 3])
            low = env.width_data[selected_band]
            width_action = maddpg.widthSelect_Agent.take_action(width_state, high, low)
            actions.append(width_action)
            next_state, rew, done, band_width_plus = env.step(state, actions)
            band_width_plus = band_width_plus.to(device)
            rew = np.array(rew)
            returns += rew / n_episode
    return returns.tolist()

import time
def train(opt):
    env = make_Env(opt.bandState_dim, opt.widthSelect_input_dim, opt.selected_bands_num, opt.file_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replay_buffer = rl_utils.ReplayBuffer(opt.buffer_size)
    t0=time.time()
    # data_idx = 0
    # rnn_input_dim = 512 + 21    # 每个波段图像提取的特征数量 + 自身21个波段
    # rnn_hidden_dim = 256
    linear_hidden_dims = [42, 10]
    critic_hidden_dim = 64
    state_dims = [21, 5]
    action_dims = [21 + 2, 1 + 1]  # 奖励系数a和b， 奖励系数# c
    critic_input_dim = sum(state_dims) + sum(action_dims)
    # band_width_plus = torch.zeros(state_dims[0], 1)
    maddpg = MADDPG(env, device, opt.actor_lr, opt.critic_lr, linear_hidden_dims,
                    critic_hidden_dim, state_dims, action_dims, critic_input_dim, opt.gamma, opt.tau)
    band_actor = maddpg.agents[0].actor
    width_actor = maddpg.agents[1].actor
    return_list = []  # 记录每一轮的回报（return）
    time_list=[]
    total_step = 0
    dones, band_width_plus, state = env.reset()  # band_width_plus: 已选波段和带宽状态
    state = state.to(device)
    band_width_plus = band_width_plus.to(device)
    for i_episode in range(opt.num_episodes):
        for e_i in range(opt.episode_length):
            print('i_episode, e_i, total_step: ',i_episode, e_i, total_step)
            if (dones[0] == True):
                dones, band_width_plus, state = env.reset()  # band_width_plus: 已选波段和带宽状态
                state = state.to(device)
                band_width_plus = band_width_plus.to(device)
            actions = []  # 两个智能体的动作的list
            band_actor_inputs = []  # band智能体actor的当前输入和下一个输入的list
            width_actor_inputs = []  # width智能体actor的当前输入和下一个输入的list
            highs = []  # width智能体actor当前和下一个输出high的list
            lows = []  # width智能体actor当前和下一个输出low的list
            states = []  # 两个智能体作为critic输入的一部分的list
            next_states = []
            band_width_pluses = []
            band_actor_inputs.append(state.detach().cpu().numpy())
            band_width_pluses.append(band_width_plus.detach().cpu().numpy())
            # band_width_plus : 已选波段和带宽的状态
            width_input, band_action = maddpg.bandSelect_Agent.take_action(state, band_width_plus, explore=True)
            states.append(width_input.squeeze().detach().cpu().numpy())
            actions.append(band_action)
            selected_band = np.argmax(band_action[:state_dims[0]])
            if (selected_band == 0 or selected_band == 20):
                width_state = torch.zeros(1, 5).to(device)
                width_state.view(-1)[2] = width_input[0][selected_band]
                high = env.width_data[selected_band]
            elif (selected_band == 1 or selected_band == 19):
                width_state = torch.zeros(1, 5).to(device)
                width_state.view(-1)[2] = width_input[0][selected_band]
                width_state.view(-1)[1] = width_input[0][selected_band - 1]
                width_state.view(-1)[3] = width_input[0][selected_band + 1]
                high = np.sum(env.width_data[selected_band - 1:selected_band + 2])
            else:
                width_state = width_input[..., (selected_band - 2):(selected_band + 3)]
                high = np.sum(env.width_data[selected_band - 2:selected_band + 3])
            low = env.width_data[selected_band]
            highs.append(high)
            lows.append(low)
            states.append(width_state.squeeze().detach().cpu().numpy())
            width_actor_inputs.append(width_state.detach().cpu().numpy())
            width_action = maddpg.widthSelect_Agent.take_action(width_state, high, low)
            actions.append(width_action)
            # actions = maddpg.take_action(state, explore=True)
            next_state, reward, dones, band_width_plus = env.step(state, actions)
            band_width_plus = band_width_plus.to(device)
            band_width_pluses.append(band_width_plus.detach().cpu().numpy())
            band_actor_inputs.append(next_state.detach().cpu().numpy())

            width_input, band_action = maddpg.bandSelect_Agent.take_action(state, band_width_plus, explore=True)
            next_states.append(width_input.squeeze().detach().cpu().numpy())
            selected_band = np.argmax(band_action[:state_dims[0]])
            if (selected_band == 0 or selected_band == 20):
                width_state = torch.zeros(1, 5).to(device)
                width_state.view(-1)[2] = width_input[0][selected_band]
                high = env.width_data[selected_band]
            elif (selected_band == 1 or selected_band == 19):
                width_state = torch.zeros(1, 5).to(device)
                width_state.view(-1)[2] = width_input[0][selected_band]
                width_state.view(-1)[1] = width_input[0][selected_band - 1]
                width_state.view(-1)[3] = width_input[0][selected_band + 1]
                high = np.sum(env.width_data[selected_band - 1:selected_band + 2])
            else:
                width_state = width_input[..., (selected_band - 2):(selected_band + 3)]
                high = np.sum(env.width_data[selected_band - 2:selected_band + 3])
            low = env.width_data[selected_band]
            highs.append(high)
            lows.append(low)
            width_actor_inputs.append(width_state.detach().cpu().numpy())
            next_states.append(width_state.squeeze().detach().cpu().numpy())
            # dones = [done, done]
            replay_buffer.add(band_actor_inputs, width_actor_inputs, states, actions, reward, next_states, dones,
                              band_width_pluses, highs, lows)
            total_step += 1
            if replay_buffer.size() >= opt.minimal_size and total_step % opt.update_interval == 0:
                sample = replay_buffer.sample(opt.batch_size)

                def stack_array(x):
                    rearranged = [[sub_x[i] for sub_x in x]
                                  for i in range(len(x[0]))]
                    return [
                        torch.FloatTensor(np.vstack(aa)).to(device)
                        for aa in rearranged
                    ]

                sample = [stack_array(x) for x in sample]
                # for a_i in range(len(env.agents)):
                for a_i in range(2):
                    maddpg.update(sample, a_i)
                maddpg.update_all_targets()
                # print("----------------------更新一次---------------------")

        if (i_episode + 1) % 100 == 0:
            ep_returns = evaluate(maddpg, n_episode=10)
            return_list.append(ep_returns)
            time_list.append(time.time()-t0)
            torch.save(band_actor.state_dict(), f'./pth_save/band{i_episode + 1}.pth')
            torch.save(width_actor.state_dict(), f'./pth_save/width{i_episode + 1}.pth')
            print(f"Episode: {i_episode + 1}, {ep_returns}")

    return_array = np.array(return_list)
    np.save('return_array',return_array)
    np.save('time_list', np.array(time_list))
    print(return_array.shape)
    for i, agent_name in enumerate(["agent_0", "agent_1"]):
        plt.figure()

        ####################
        # plt.plot(
        #     np.arange(return_array.shape[0]) * 100,
        #     rl_utils.moving_average(return_array[:, i], 9))
        ####################

        plt.plot(np.arange(return_array.shape[0]) * 100,return_array[:, i])
        plt.xlabel("Episodes")
    plt.show()


if __name__ == '__main__':
    opt = get_args()
    train(opt)