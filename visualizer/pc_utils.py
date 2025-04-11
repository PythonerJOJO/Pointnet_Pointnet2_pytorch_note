""" 处理点云的实用函数。
作者: Charles R. Qi, Hao Su
日期: 2016年11月
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# 绘制点云
from visualizer.eulerangles import euler2mat

# 点云输入输出
import numpy as np
from visualizer.plyfile import PlyData, PlyElement

# ----------------------------------------
# 点云/体素转换
# ----------------------------------------

def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """ 输入为BxNx3的点云批次
        输出为Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b, :, :]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


def point_cloud_to_volume(points, vsize, radius=1.0):
    """ 输入为Nx3的点。
        输出为vsize*vsize*vsize
        假设点的范围在[-radius, radius]内
    """
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol


# a = np.zeros((16,1024,3))
# print(point_cloud_to_volume_batch(a, 12, 1.0, False).shape)

def volume_to_point_cloud(vol):
    """ vol是大小为vsize*vsize*vsize的占用网格（值为0或1）
        返回Nx3的numpy数组。
    """
    vsize = vol.shape[0]
    assert (vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a, b, c] == 1:
                    points.append(np.array([a, b, c]))
    if len(points) == 0:
        return np.zeros((0, 3))
    points = np.vstack(points)
    return points


# ----------------------------------------
# 点云输入输出
# ----------------------------------------

def read_ply(filename):
    """ 从文件名对应的PLY文件中读取XYZ点云 """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


def write_ply(points, filename, text=True):
    """ 输入为Nx3，将点以PLY格式写入文件名对应的文件。 """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


# ----------------------------------------
# 简单的点云和体素渲染器
# ----------------------------------------

def draw_point_cloud(input_points, canvasSize=500, space=200, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0, 1, 2], normalize=True):
    """ 将点云渲染为带有Alpha通道的图像。
        输入:
            points: Nx3的numpy数组（+y为向上方向）
        输出:
            大小为canvasSizexcanvasSize的灰度图像，以numpy数组形式呈现
    """
    image = np.zeros((canvasSize, canvasSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # 归一化点云
    # 我们将点云缩放到单位球内
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
        points /= furthest_distance

    # 预计算高斯圆盘
    radius = (diameter - 1) / 2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i - radius) + (j - radius) * (j - radius) <= radius * radius:
                disk[i, j] = np.exp((-(i - radius) ** 2 - (j - radius) ** 2) / (radius ** 2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]

    # 按z缓冲区对点进行排序
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])

    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize / 2 + (x * space)
        yc = canvasSize / 2 + (y * space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc
        py = dy + yc

        image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3

    image = image / np.max(image)
    return image


def point_cloud_three_views(points):
    """ 输入为Nx3的numpy数组点云（+y为向上方向）。
        返回大小为500x1500的灰度图像numpy数组。 """
    # +y为向上方向
    # xrot是方位角
    # yrot是平面内旋转
    # zrot是仰角
    img1 = draw_point_cloud(points, zrot=110 / 180.0 * np.pi, xrot=45 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    img2 = draw_point_cloud(points, zrot=70 / 180.0 * np.pi, xrot=135 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    img3 = draw_point_cloud(points, zrot=180.0 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    image_large = np.concatenate([img1, img2, img3], 1)
    return image_large


from PIL import Image


def point_cloud_three_views_demo():
    """ draw_point_cloud函数的演示 """
    DATA_PATH = '../data/ShapeNet/'
    train_data, _, _, _, _, _ = load_data(DATA_PATH, classification=False)
    points = train_data[1]
    im_array = point_cloud_three_views(points)
    img = Image.fromarray(np.uint8(im_array * 255.0))
    img.save('example.jpg')


if __name__ == "__main__":
    from data_utils.ShapeNetDataLoader import load_data
    point_cloud_three_views_demo()

import matplotlib.pyplot as plt


def pyplot_draw_point_cloud(points, output_filename):
    """ points是Nx3的numpy数组 """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # savefig(output_filename)


def pyplot_draw_volume(vol, output_filename):
    """ vol的大小为vsize*vsize*vsize
        将图像输出到output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)