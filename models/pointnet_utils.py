import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    """
    3*3 的 T-Net transform : 类似一个 mini-PointNet
    """

    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        # 批归一化
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Symmetric function: max pooling 最大池化得到全局特征
        x = torch.max(x, 2, keepdim=True)[0]
        # 参数拉直(展平)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # 9个元素

        # 展平的对角矩阵
        iden = (
            Variable(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
                )
            )
            .view(1, 9)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden  # affine transformation 仿射变换
        x = x.view(-1, 3, 3)  # 用 view reshape
        return x


class STNkd(nn.Module):
    """
    默认作 64*64 的变换
    """

    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    """
    对于分类网络：编码器完成从 input points(n*3) 到提取 global feature(1024)
    对于分割网络：编码器还包括 local embedding(n*64) 和 global feature(1024) 的拼接
    """

    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        """
        1. 提取中心点；
        2. 做类似卷积，把邻近的点联系起来
        """
        # batchsize,3(xyz坐标) 或 6(xyz+法向量), 1024(一个物体所取的点数目)
        B, D, N = x.size()

        trans = self.stn(x)
        x = x.transpose(2, 1)  # 交换 tensor 的其中两个维度
        if D > 3:
            feature = x[:, :, 3:]  # 先把 法向量 拿出来
            x = x[:, :, :3]  # 再把 xyz 拿出来
        x = torch.bmm(x, trans)  # 两个三维张量矩阵相乘
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)

        # MLP 网络
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x  # 局部特征

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:  # 在分割任务中，需要额外返回 全局特征 与 局部特征 的拼接
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    """
    对 特征转换矩阵 做正则化

    让 trans transformation matrix 接近于正交矩阵，就不会损失特征信息
    """
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]  # 2维单位矩阵
    if trans.is_cuda:
        I = I.cuda()

    # 正则化损失函数    (A*A^T - I) 的 L2 范数, 即 Frobenius 范数，再求其均值
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))
    )
    return loss
