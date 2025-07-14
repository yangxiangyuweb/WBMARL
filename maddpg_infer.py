'''
特征提取  Resnet101、convlstm
波段数 21、42、84
'''

import os
import sys
sys.path.append('../train')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def read_hyper(filepath):
    band = int(len(os.listdir(filepath))/2)
    hyper1 = np.zeros([256, 320, band], dtype=np.float32)
    hyper0 = np.zeros([256, 320, band], dtype=np.float32)
    i = 0
    j = 0
    for imgname in os.listdir(filepath):
        if imgname.find('_0') != -1:
            imgpath = os.path.join(filepath, imgname)
            img = cv2.imread(imgpath, -1)
            img = np.flip(img)
            hyper1[:, :, i] = img
            i = i+1
        else:
            imgpath = os.path.join(filepath, imgname)
            img = cv2.imread(imgpath, -1)
            img = np.flip(img)
            hyper0[:, :, j] = img
            j = j + 1
    hyper = hyper1-hyper0
    hyper[hyper<0]=0
    return hyper

def read_hyper1(filepath):
    imglist=[]
    if filepath.find('sky')>=0:
        qianzhui='sky'
    if filepath.find('road')>=0:
        qianzhui='road'
    if filepath.find('forest')>=0:
        qianzhui='forest'
    if filepath.find('building')>=0:
        qianzhui='building'
    for i in range(len(os.listdir(filepath))):
        imgname=qianzhui+str(i)+'.tif'
        imgpath = os.path.join(filepath, imgname)
        print(imgpath)
        hyper = cv2.imread(imgpath, -1)
        imglist.append(hyper)
    imglist=np.array(imglist)
    return imglist


# ConvLSTM
class ConvLSTMCell(nn.Module):
    # 这里面全都是数，衡量后面输入数据的维度/通道尺寸
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 卷积核为一个数组
        self.kernel_size = kernel_size
        # 填充为高和宽分别填充的尺寸
        self.padding_size = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(self.input_dim + self.hidden_dim,
                              4 * self.hidden_dim,  # 4* 是因为后面输出时要切4片
                              self.kernel_size,
                              padding=self.padding_size,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat((input_tensor, h_cur), dim=1)
        combined_conv = self.conv(combined)
        cc_f, cc_i, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # torch.sigmoid(),激活函数--
        # nn.functional中的函数仅仅定义了一些具体的基本操作，
        # 不能构成PyTorch中的一个layer
        # torch.nn.Sigmoid()(input)等价于torch.sigmoid(input)
        f = torch.sigmoid(cc_f)
        i = torch.sigmoid(cc_i)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # 这里的乘是矩阵对应元素相乘，哈达玛乘积
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return c_next, h_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        # 返回两个是因为cell的尺寸与h一样
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=False,
                 return_all_layers=False):
        super(ConvLstm, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # 为了储存每一层的参数尺寸
        cell_list = []
        for i in range(0, num_layers):
            # 注意这里利用lstm单元得出到了输出h，h再作为下一层的输入，依次得到每一层的数据维度并储存
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size,
                                          bias=self.bias
                                          ))
        # 将上面循环得到的每一层的参数尺寸/维度，储存在self.cell_list中，后面会用到
        # 注意这里用了ModuLelist函数，模块化列表
        self.cell_list = nn.ModuleList(cell_list)

    # 这里forward有两个输入参数，input_tensor 是一个五维数据
    # （t时间步,b输入batch_ize,c输出数据通道数--维度,h,w图像高乘宽）
    # hidden_state=None 默认刚输入hidden_state为空，等着后面给初始化
    def forward(self, input_tensor, hidden_state=None):
        # 先调整一下输出数据的排列
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        # 取出图片的数据，供下面初始化使用
        b, _, _, h, w = input_tensor.size()
        # 初始化hidd_state,利用后面和lstm单元中的初始化函数
        hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        # 储存输出数据的列表
        layer_output_list = []
        layer_state_list = []

        seq_len = input_tensor.size(1)

        # 初始化输入数据
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                # 每一个时间步都更新 h,c
                # 注意这里self.cell_list是一个模块(容器)
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                # 储存输出，注意这里 h 就是此时间步的输出
                output_inner.append(h)

            # 这一层的输出作为下一次层的输入,
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            # 储存每一层的状态h，c
            layer_state_list.append([h, c])

        # 选择要输出所有数据，还是输出最后一层的数据
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            layer_state_list = layer_state_list[-1:]

        return layer_output_list, layer_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

# 波段选择actor
class BandSelect_Actor(nn.Module):
    def __init__(self, state_size, action_dim,  linear_hidden_dim):
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

    def forward(self, x, band_width_plus):
        x = self.conv(x)
        x = x.squeeze()
        plused_tensor = torch.cat((x, band_width_plus), dim=-1)
        x = F.relu(self.fc1(plused_tensor))
        x = F.relu(self.fc2(x))
        x = x.squeeze(-1)
        if x.ndim == 1:
            x = torch.unsqueeze(x, dim=0)
        x1 = F.relu(self.fc3(x))
        x1 = self.fc4(x1)
        x1 = torch.cat((x1[:, :self.state_size], torch.softmax(x1[:, self.state_size:], dim=-1)), dim=-1)

        return x, x1

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
        action = torch.cat(((output_modified / 2 * (high - low) + low),
                            torch.sigmoid(x[:, 1:])), dim=1)
        return action

start_time = time.time()
input_channels = 1
hidden_channels = [5, 5]
kernel_size = (3, 3)
num_layers = 2
batch_first = True
bias = True
return_all_layers = False
conv_lstm = ConvLstm(input_channels, hidden_channels, kernel_size, num_layers, batch_first, bias, return_all_layers)
conv_lstm.load_state_dict(torch.load('../model/static_pth/conv_lstm_multy.pth'))

state_dims = [21, 5]
action_dims = [21 + 2, 1 + 1]  # 奖励系数a和b， 奖励系数c
linear_hidden_dims = [42, 10]
select_bands = 5
selected_band = 0
start_time3 = time.time()
'''band_actor和width_actor用时'''
band_actor = BandSelect_Actor(state_dims[0], action_dims[0], linear_hidden_dims[0])
width_actor = Bandwidth_Actor(state_dims[1], action_dims[1], linear_hidden_dims[1])
end_time3 = time.time()
# band_actor.load_state_dict(torch.load('../model/static_pth/bandselect.pth'))
# width_actor.load_state_dict(torch.load('../model/static_pth/widthselect.pth'))




width_data = np.array([51.93, 49.47, 47.72, 46.09, 44.94, 44.2, 43.26, 42.51, 41.7,
                40.83, 39.89, 38.89, 37.86, 36.81, 36.1, 35.06, 34.03, 33.37, 32.41, 31.49, 30.9])  # 每个波段的带宽
wavelength_data=np.array([4808.036931622306, 4742.117291439937, 4677.428858401089, 4613.958667157539, 4551.69375236106,
                          4490.621148663428, 4430.7278907164155, 4372.001013171803, 4314.427550681357, 4257.994537896858,
                          4202.68900947008, 4148.498000052796, 4095.4085442967835, 4043.4076768538152, 3992.4824323756675,
                          3942.619845514114, 3893.80695092093, 3846.03078324789, 3799.2783771467703, 3753.5367672693437,
                          3708.792988267387])
# 21, 5, 256, 320

band_actor.load_state_dict(torch.load('../train/pth_save/sky/band900.pth'))
width_actor.load_state_dict(torch.load('../train/pth_save/sky/width900.pth'))

'''1. forest 2. road 3. building 4. sky 5. plane'''
file_idx = 4
if(file_idx == 1):
    filepath = r'D:\yangxiangyu\PythonWorks\maddpg_bandwidthSelect\MSI\building\2'
elif(file_idx == 2):
    filepath = r'D:\yangxiangyu\PythonWorks\maddpg_bandwidthSelect\MSI\forest\1'
elif(file_idx == 3):
    filepath = r'D:\yangxiangyu\PythonWorks\maddpg_bandwidthSelect\MSI\road\1'
elif(file_idx == 4):
    filepath = r'D:\yangxiangyu\PythonWorks\maddpg_bandwidthSelect\MSI\sky\1'
elif(file_idx == 5):
    filepath = r'D:\yangxiangyu\Data\AOTF_Bandselect\sky_4ms\1'

input_data = torch.from_numpy(read_hyper1(filepath).reshape((1, 21, 1, 256, 320)))
multispec_feature, state_10 = conv_lstm(input_data)

# multispec_feature=np.load(r'../train/feature_data_2/building/feature2.npy')
# multispec_feature=np.expand_dims(multispec_feature,axis=0)
# multispec_feature= torch.tensor(multispec_feature)
# multispec_feature=[multispec_feature]

band_width_selected = torch.zeros(21, 1)
res = np.zeros((2, select_bands))
select_result=res.copy()
for i in range(select_bands):
    start_time1 = time.time()
    width_input, action = band_actor(torch.squeeze(multispec_feature[0], dim=0), band_width_selected)
    end_time1 = time.time()
    for j in res[0]:
        action[0][int(j)] = 0

    selected_band = np.argmax(action[:, :21].detach().numpy())
    res[0][i] = selected_band
    select_result[0][i] = wavelength_data[int(res[0][i])]
    if (selected_band == 0 or selected_band == 20):
        width_state = torch.zeros(1, 5)
        width_state.view(-1)[2] = width_input[0][selected_band]
        high = width_data[selected_band]
    elif (selected_band == 1 or selected_band == 19):
        width_state = torch.zeros(1, 5)
        width_state.view(-1)[2] = width_input[0][selected_band]
        width_state.view(-1)[1] = width_input[0][selected_band - 1]
        width_state.view(-1)[3] = width_input[0][selected_band + 1]
        high = np.sum(width_data[selected_band - 1:selected_band + 2])
    else:
        width_state = width_input[..., (selected_band - 2):(selected_band + 3)]
        high = np.sum(width_data[selected_band - 2:selected_band + 3])
    low = width_data[selected_band]
    high_tensor = torch.tensor([high], dtype=torch.float32)
    low_tensor = torch.tensor([low], dtype=torch.float32)
    start_time2 = time.time()
    width_action = width_actor(width_state, high=high_tensor, low=low_tensor)
    end_time2 = time.time()
    width_action = width_action[0][0].item()
    res[1][i] = width_action
    select_result[1][i] = res[1][i]

    if (width_action > width_data[selected_band] - 1 and width_action < width_data[selected_band] + 1):
        band_width_selected[selected_band] += 1
    elif (width_action >= width_data[selected_band] + 1 and width_action < np.sum(width_data[selected_band - 1: selected_band + 2])):
        n = (width_action - width_data[selected_band]) / (width_data[selected_band - 1] + width_data[selected_band + 1])
        band_width_selected[selected_band] += 1
        band_width_selected[selected_band - 1] += n
        band_width_selected[selected_band + 1] += n
    else:

        n = (width_action - np.sum(width_data[selected_band - 1: selected_band + 2])) / (
                     width_data[selected_band - 2] + width_data[selected_band + 2])
        band_width_selected[selected_band] += 1
        band_width_selected[selected_band - 1] += 1
        band_width_selected[selected_band + 1] += 1
        band_width_selected[selected_band - 2] += n
        band_width_selected[selected_band + 2] += n

def onehot_from_logits(logits, eps = 0.01):
    '''生成最优动作的独热（one-hot）形式'''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作，转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size = logits.shape[0])]], requires_grad=False)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])

wavelengths_selected_index=np.array(res[0])[np.argsort(-1*res[0])]
bandwidths_selected=select_result[1,np.argsort(select_result[0,:])]
wavelengths_selected=np.sort(select_result[0,:])

print(wavelengths_selected_index)
print(wavelengths_selected)
print(bandwidths_selected)



