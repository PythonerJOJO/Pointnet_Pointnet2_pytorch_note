import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(
        self,
        k=40,  # 要分类的物体的类别的数量
        normal_channel=True,  # 使用法向量
    ):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        # 获取特征输出
        self.feat = PointNetEncoder(
            global_feat=True, feature_transform=True, channel=channel
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 分类模型，x是 1024，返回的全局特征(global feature)
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        # nll_loss 输入对数概率向量和一个目标标签，适合log_softmax
        loss = F.nll_loss(pred, target)  # 分类损失，比对预测值和标签值
        # 特征变换正则化损失
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        # 在总损失中，mat_diff_loss的权重变小，强调分类的损失
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
