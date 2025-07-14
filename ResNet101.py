import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import functional as F
import os

class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet101FeatureExtractor, self).__init__()

        # 加载预训练的ResNet101
        resnet101 = models.resnet101(pretrained=False)
        resnet101.load_state_dict(torch.load('resnet101-63fe2227.pth'))

        # 修改第一层卷积以接受单通道输入
        original_conv1 = resnet101.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 初始化新卷积层权重(使用原RGB权重的平均值)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

        # 提取ResNet的前四个层(去掉最后的全连接层)
        self.feature_extractor = nn.Sequential(
            self.conv1,
            resnet101.bn1,
            resnet101.relu,
            resnet101.maxpool,
            resnet101.layer1,
            resnet101.layer2,
            resnet101.layer3,
            resnet101.layer4
        )

        # 添加转置卷积层将特征图上采样回原始尺寸
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 5, kernel_size=1))  # 最终输出5通道

        # 冻结ResNet部分的参数(可选)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 输入形状: (batch, 1, 320, 256)

        # 标准化输入(使用ImageNet统计量)
        x = x.repeat(1, 3, 1, 1)  # 临时复制为3通道用于标准化
        x = F.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        x = x[:, :1, :, :]  # 取回单通道

        # 提取特征
        features = self.feature_extractor(x)  # 输出形状: (batch, 2048, 10, 8)

        # 上采样回原始尺寸
        output = self.upsample(features)  # 输出形状: (batch, 5, 320, 256)

        return output

import numpy as np
import cv2

def read_hyper1(filepath):
    imglist=[]
    for img in os.listdir(filepath):
        imgpath = os.path.join(filepath, img)
        hyper = cv2.imread(imgpath, -1)
        imglist.append(hyper)
    imglist=np.array(imglist)
    imglist=np.expand_dims(imglist,axis=1)
    return imglist
# 测试代码

if __name__ == "__main__":
    # 创建模型实例
    model = ResNet101FeatureExtractor(pretrained=True).eval()
    filedir = r''
    dstdir = r''
    for name in os.listdir(filedir):
        filepath = os.path.join(filedir, name)
        dstpath = os.path.join(dstdir, 'feature' + str(int(len(os.listdir(dstdir))) + int(name)))
        input_tensor = torch.tensor(read_hyper1(filepath).reshape(( 21, 1, 256, 320)),dtype=torch.float32)  # 添加batch维度
        # 前向传播
        with torch.no_grad():
            output = model(input_tensor)
        # 保存输出特征
        multispec_feature = output.numpy()
        np.save(dstpath, multispec_feature)
        print("输入形状:", input_tensor.shape)
        print("输出形状:", output.shape)
        print("multispec_feature:", multispec_feature.shape)
