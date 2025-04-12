import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

"""
PointNet++ 中有一些关键的处理
    farthest_point_sample 最远点采样
    query_ball_point 球查询
    sample_and_group 最远点采样和球查询
    sample_and_group_all 只有一个 group
    
    pc_normalize 归一化点云
    square_distance 求欧氏距离
    index_points 点云查找
"""


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    """
    归一化点云
        以 centroid 为中心，球半径为1

    :param pc: 点云
    """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)  # 质心
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # 球半径
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    计算每两个点之间的欧几里得距离。

    理论:
        输入两组点，
            N 为第一组点 src 的个数，
            M 为第二组点 dst 的个数，
            C 为输入点的通道数(输入 xyz 时，通道数为3)


    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    :param    src: 源点，[B, N, C]
    :param    dst: 目标点，[B, M, C]
    :return dist:
        两组点两两之间的欧几里德距离，[B, N, M],其中 B 为 batch_size
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    根据输入的点云数据和索引，返回索引的点云数据
    例如:
        输入 points = [B,2048,3] idx = [5,666,1000,2000]
        表示要找出 每一个 batch 中的 第 5、666、1000、2000 个 点云数据集

    :param  points: 输入的点云数据，[B, N, C], [batch_size,点云数量，通道数]
    :param  idx: 采样索引数据，[B, S]
    :return new_points: 索引后的点云数据，[B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    最远点采样
        从输入的点云数据中，按照所需要的点的个数 npoint 采样出足够多的点
        因为是最远点采样，所以点与点之间的距离要尽量的远

    :param  xyz: 点云数据，[B, N, 3]
    :param  npoint: 采样点数量
    :return centroids: npoint 个采样点云索引，[B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    输入:
        radius: 局部区域半径
        nsample: 局部区域内的最大采样数量
        xyz: 所有点，[B, N, 3]
        new_xyz: 查询点，[B, S, 3]
    返回:
        group_idx: 分组点索引，[B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = (
        torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    )
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    最远点采用和球查询
        用于将整个点云分散成局部的 group
            对每个 group，都可以用 PointNet 单独的提取局部的全局特征


    :param  npoint: 要找多少个中心点
    :param  radius: 半径
    :param  nsample: 在点云中找多少个点
    :param  xyz: 输入点位置数据，[B, N, 3]
    :param  points: 输入点数据，[B, N, D]

    :return new_xyz: 采样点位置数据，[B, npoint, nsample, 3]
    :return new_points: 采样点数据，[B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=-1
        )  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    输入:
        xyz: 输入点位置数据，[B, N, 3]
        points: 输入点数据，[B, N, D]
    返回:
        new_xyz: 采样点位置数据，[B, 1, 3]
        new_points: 采样点数据，[B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """
    set abstraction 操作
    1. 通过 sample_and_group 的操作，形成局部 group
    2. 对局部 group 中的每一个 点 做 MLP 操作
    3. 进行局部的最大池化，得到局部的全局特征
    """

    def __init__(
        self,
        npoint: int,  # 在点云数据中进行最远点采样的点的个数
        radius: float,  # 每个 质心点 的半径
        nsample: int,  # 每个局部区域采样的点数
        in_channel,  #
        mlp: list[int],  # 多层感知机，例如 [128,128,256]
        group_all: bool,
    ):
        # super(PointNetSetAbstraction, self).__init__()
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        输入:
            xyz: 输入点位置数据，[B, C, N]
            points: 输入点数据，[B, D, N]
        返回:
            new_xyz: 采样点位置数据，[B, C, S]
            new_points_concat: 采样点特征数据，[B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:  # 形成局部 group
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        # new_xyz: 采样点位置数据，[B, npoint, C]
        # new_points: 采样点数据，[B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        # 对每个局部点做 MLP 操作
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """
    Msg 在 radius_list 输入的是一个 list ，例如[0.1,0.2,0.4]
    对于不同的半径做 ball_query, 将不同半径下的点云特征保存到 new_points_list 中，最后拼接到一起

    """

    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        # super(PointNetSetAbstractionMsg, self).__init__()
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor):
        """
        输入:
            xyz: 输入点位置数据，[B, C, N]
            points: 输入点数据，[B, D, N]
        返回:
            new_xyz: 采样点位置数据，[B, C, S]
            new_points_concat: 采样点特征数据，[B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # 交换第1、2维的位置， shape(16,1024,3)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    """
    用作在分割任务时做上采样
    主要通过线性插值和 MLP 来完成
    """

    def __init__(self, in_channel, mlp):
        # super(PointNetFeaturePropagation, self).__init__()
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: torch.Tensor,
        points2: torch.Tensor,
    ):
        """
        输入:
            xyz1: 输入点位置数据，[B, C, N]
            xyz2: 采样后的输入点位置数据，[B, C, S]
            points1: 输入点数据，[B, D, N]
            points2: 输入点数据，[B, D, S]
        返回:
            new_points: 上采样后的点数据，[B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
