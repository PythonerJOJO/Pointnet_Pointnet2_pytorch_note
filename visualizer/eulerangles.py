# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   请查看与NiBabel包一起分发的COPYING文件，以了解版权和许可条款。
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' 实现欧拉角旋转及其转换的模块
参考：
* http://en.wikipedia.org/wiki/Rotation_matrix
* http://en.wikipedia.org/wiki/Euler_angles
* http://mathworld.wolfram.com/EulerAngles.html
另请参阅：James Diebel 所著的 *《用欧拉角和四元数表示姿态：参考》* (2006)。最后一次找到的缓存 PDF 链接如下：
http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.110.5134
欧拉旋转定理表明，三维空间中的任何旋转都可以用三个角度来描述。我们将这三个角度称为 *欧拉角向量*，并将向量中的角度分别记为 :math:`alpha`、:math:`beta` 和 :math:`gamma`。该向量表示为 [ :math:`alpha`, :math:`beta`, :math:`gamma` ]，在本描述中，参数的顺序指定了旋转发生的顺序（即与 :math:`alpha` 对应的旋转首先执行）。
为了明确 *欧拉角向量* 的含义，我们需要指定与 :math:`alpha`、:math:`beta` 和 :math:`gamma` 对应的旋转所围绕的轴。
因此，对于 :math:`alpha`、:math:`beta` 和 :math:`gamma` 这三个旋转，分别有三个轴，我们将它们称为 :math:`i`、:math:`j`、:math:`k`。
我们将绕轴 `i` 旋转 :math:`alpha` 表示为一个 3x3 的旋转矩阵 `A`。类似地，绕轴 `j` 旋转 :math:`beta` 得到 3x3 矩阵 `B`，绕轴 `k` 旋转 :math:`gamma` 得到矩阵 `G`。那么，由欧拉角向量 [ :math:`alpha`, :math:`beta`, :math:`gamma` ] 表示的整个旋转 `R` 可以通过以下公式计算：
   R = np.dot(G, np.dot(B, A))
参考：http://mathworld.wolfram.com/EulerAngles.html
顺序 :math:`G B A` 表示旋转按照向量中的顺序执行（即先绕轴 `i` 旋转 :math:`alpha`，对应矩阵 `A`）。
要将给定的欧拉角向量转换为有意义的旋转和旋转矩阵，我们需要定义以下内容：
* 轴 `i`、`j`、`k`
* 旋转矩阵是应用于待变换向量的左侧（向量为列向量）还是右侧（向量为行向量）
* 旋转时轴是否随旋转移动（内旋）——与轴固定而向量在轴坐标系内移动的情况（外旋）进行对比
* 坐标系的手性
参考：http://en.wikipedia.org/wiki/Rotation_matrix#Ambiguities
我们采用以下约定：
* 轴 `i`、`j`、`k` 分别为 `z`、`y` 和 `x` 轴。因此，在我们的约定中，欧拉角向量 [ :math:`alpha`, :math:`beta`, :math:`gamma` ] 表示先绕 `z` 轴旋转 :math:`alpha` 弧度，接着绕 `y` 轴旋转 :math:`beta` 弧度，最后绕 `x` 轴旋转 :math:`gamma` 弧度。
* 旋转矩阵应用于左侧，右侧为列向量。因此，如果 `R` 是旋转矩阵，`v` 是一个 3xN 的矩阵，其中包含 N 个列向量，那么变换后的向量集 `vdash` 可以通过以下公式计算：
  ``vdash = np.dot(R, v)``
* 外旋——轴是固定的，不会随旋转而移动。
* 右手坐标系
绕 ``z`` 轴、``y`` 轴、``x`` 轴的旋转约定（即先绕 ``z`` 轴，再绕 ``y`` 轴，最后绕 ``x`` 轴）也被（容易混淆地）称为 "xyz"、俯仰 - 横滚 - 偏航角、卡尔丹角或泰特 - 布莱恩角。
'''

import math

import sys
if sys.version_info >= (3,0):
    from functools import reduce

import numpy as np


_FLOAT_EPS_4 = np.finfo(float).eps * 4.0


def euler2mat(z=0, y=0, x=0):
    ''' 返回绕 z、y 和 x 轴旋转的矩阵
    使用上述先绕 z 轴，再绕 y 轴，最后绕 x 轴的约定
    参数
    ----------
    z : 标量
       绕 z 轴的旋转角度（以弧度为单位，首先执行）
    y : 标量
       绕 y 轴的旋转角度（以弧度为单位）
    x : 标量
       绕 x 轴的旋转角度（以弧度为单位，最后执行）
    返回
    -------
    M : 形状为 (3, 3) 的数组
       与给定角度产生相同旋转效果的旋转矩阵
    示例
    --------
    >>> zrot = 1.3  # 弧度
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True
    输出的旋转矩阵等于各个单独旋转的组合
    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True
    你可以通过命名参数指定旋转
    >>> np.all(M3 == euler2mat(x=xrot))
    True
    当将 M 应用于向量时，向量应该是位于 M 右侧的列向量。如果右侧是一个二维数组而不是向量，那么该二维数组的每一列都代表一个向量。
    >>> vec = np.array([1, 0, 0]).reshape((3, 1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0], [0, 1, 0]]).T  # 得到一个 3x2 的数组
    >>> vecs2 = np.dot(M, vecs)
    旋转方向为逆时针。
    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    True
    注意事项
    -----
    旋转方向由右手定则确定（将右手拇指沿旋转轴方向放置，拇指指向轴的正方向；然后卷曲手指，手指卷曲的方向就是旋转的方向）。因此，从旋转轴的正方向向负方向看，旋转是逆时针的。
    '''
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def mat2euler(M, cy_thresh=None):
    ''' 从 3x3 矩阵中推导欧拉角向量
    使用上述约定。
    参数
    ----------
    M : 类数组对象，形状为 (3, 3)
    cy_thresh : 可选，None 或标量
       当低于该阈值时，放弃使用直接的反正切函数来估计 x 旋转。如果为 None（默认值），则根据输入的精度进行估计。
    返回
    -------
    z : 标量
    y : 标量
    x : 标量
       分别为绕 z、y、x 轴的旋转角度（以弧度为单位）
    注意事项
    -----
    如果没有数值误差，该函数可以通过 Sympy 推导出的先绕 z 轴，再绕 y 轴，最后绕 x 轴的旋转矩阵表达式来实现，该表达式为：
      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    由此可以直接推导出 z、y 和 x 的表达式：
       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)
    当 cos(y) 接近零时会出现问题，因为以下两个式子：
       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    都会接近 atan2(0, 0)，从而导致结果非常不稳定。
    下面用于解决数值不稳定问题的 ``cy`` 方法来自：*《图形宝石 IV》*，Paul Heckbert（编辑），学术出版社，1994 年，ISBN: 0123361559。具体来说，它来自 Ken Shoemake 的 EulerAngles.c 文件，用于处理 cos(y) 接近零的情况：
    参考：http://www.graphicsgems.org/
    从网站上的信息来看，该代码的许可方式为 "可以无限制使用"。
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh:  # cos(y) 不接近零，采用标准形式
        z = math.atan2(-r12,  r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy)  # atan2(sin(y), cy)
        x = math.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else:  # cos(y) 接近零，因此 x -> 0.0（见上文）
        # 此时 r21 -> sin(z)，r22 -> cos(z)
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy)  # atan2(sin(y), cy)
        x = 0.0
    return z, y, x


def euler2quat(z=0, y=0, x=0):
    ''' 返回与这些欧拉角对应的四元数
    使用上述先绕 z 轴，再绕 y 轴，最后绕 x 轴的约定
    参数
    ----------
    z : 标量
       绕 z 轴的旋转角度（以弧度为单位，首先执行）
    y : 标量
       绕 y 轴的旋转角度（以弧度为单位）
    x : 标量
       绕 x 轴的旋转角度（以弧度为单位，最后执行）
    返回
    -------
    quat : 形状为 (4,) 的数组
       四元数，格式为 w, x, y, z（实部在前，向量部分在后）
    注意事项
    -----
    我们可以在 Sympy 中通过以下步骤推导出这个公式：
    1. 绕任意轴旋转 theta 弧度对应的四元数公式：
       http://mathworld.wolfram.com/EulerParameters.html
    2. 根据 1.) 生成绕 ``x``、``y``、``z`` 轴旋转 theta 弧度对应的四元数公式
    3. 应用四元数乘法公式 -
       http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - 到 2.) 中的公式，得到组合旋转的公式。
    '''
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
             cx*cy*cz - sx*sy*sz,
             cx*sy*sz + cy*cz*sx,
             cx*cz*sy - sx*cy*sz,
             cx*cy*sz + sx*cz*sy])


def quat2euler(q):
    ''' 返回与四元数 `q` 对应的欧拉角
    参数
    ----------
    q : 包含 4 个元素的序列
       四元数的 w, x, y, z
    返回
    -------
    z : 标量
       绕 z 轴的旋转角度（以弧度为单位，首先执行）
    y : 标量
       绕 y 轴的旋转角度（以弧度为单位）
    x : 标量
       绕 x 轴的旋转角度（以弧度为单位，最后执行）
    注意事项
    -----
    可以通过结合 ``quat2mat`` 和 ``mat2euler`` 函数的部分内容来稍微减少计算量，但计算量的减少很小，而且代码重复度很高。
    '''
    # 延迟导入以避免循环依赖
    import nibabel.quaternions as nq
    return mat2euler(nq.quat2mat(q))


def euler2angle_axis(z=0, y=0, x=0):
    ''' 返回与这些欧拉角对应的旋转角度和旋转轴
    使用上述先绕 z 轴，再绕 y 轴，最后绕 x 轴的约定
    参数
    ----------
    z : 标量
       绕 z 轴的旋转角度（以弧度为单位，首先执行）
    y : 标量
       绕 y 轴的旋转角度（以弧度为单位）
    x : 标量
       绕 x 轴的旋转角度（以弧度为单位，最后执行）
    返回
    -------
    theta : 标量
       旋转角度
    vector : 形状为 (3,) 的数组
       旋转所围绕的轴
    示例
    --------
    >>> theta, vec = euler2angle_axis(0, 1.5, 0)
    >>> print(theta)
    1.5
    >>> np.allclose(vec, [0, 1, 0])
    True
    '''
    # 延迟导入以避免循环依赖
    import nibabel.quaternions as nq
    return nq.quat2angle_axis(euler2quat(z, y, x))


def angle_axis2euler(theta, vector, is_normalized=False):
    ''' 将旋转角度和旋转轴对转换为欧拉角
    参数
    ----------
    theta : 标量
       旋转角度
    vector : 包含 3 个元素的序列
       指定旋转轴的向量。
    is_normalized : 可选，布尔值
       如果向量已经归一化（模为 1），则为 True。默认值为 False
    返回
    -------
    z : 标量
    y : 标量
    x : 标量
       分别为绕 z、y、x 轴的旋转角度（以弧度为单位）
    示例
    --------
    >>> z, y, x = angle_axis2euler(0, [1, 0, 0])
    >>> np.allclose((z, y, x), 0)
    True
    注意事项
    -----
    可以通过结合 ``angle_axis2mat`` 和 ``mat2euler`` 函数的部分内容来稍微减少计算量，但计算量的减少很小，而且代码重复度很高。
    '''
    # 延迟导入以避免循环依赖
    import nibabel.quaternions as nq
    M = nq.angle_axis2mat(theta, vector, is_normalized)
    return mat2euler(M)
    