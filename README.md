# PointNet 和 PointNet++ 的 PyTorch 实现

此仓库是使用 PyTorch 对 [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) 和 [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) 的实现。

## 更新情况
**2021/03/27**：
(1) 发布语义分割的预训练模型，其中 PointNet++ 能达到 **53.5%** 的平均交并比（mIoU）。
(2) 在 `log/` 目录下发布分类和部件分割的预训练模型。

**2021/03/20**：更新分类代码，包括：
(1) 增加对 **ModelNet10** 数据集的训练代码。使用 `--num_category 10` 进行设置。
(2) 增加仅在 CPU 上运行的代码。使用 `--use_cpu` 进行设置。
(3) 增加离线数据预处理代码以加速训练。使用 `--process_data` 进行设置。
(4) 增加使用均匀采样进行训练的代码。使用 `--use_uniform_sample` 进行设置。

**2019/11/26**：
(1) 修正了之前代码中的一些错误，并添加了数据增强技巧。现在仅用 1024 个点进行分类，准确率可达 **92.8%**！
(2) 添加了测试代码，包括分类、分割以及带可视化的语义分割。
(3) 将所有模型整理到 `./models` 文件中，方便使用。

## 安装
最新代码在 Ubuntu 16.04、CUDA 10.1、PyTorch 1.6 和 Python 3.7 环境下测试通过：
```shell
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
```

## 分类（ModelNet10/40）
### 数据准备
从[此处](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)下载对齐后的 **ModelNet** 数据集，并保存到 `data/modelnet40_normal_resampled/` 目录下。

### 运行
你可以使用以下代码运行不同模式。
* 若要使用数据的离线处理功能，首次运行时可使用 `--process_data`。你也可以从[此处](https://drive.google.com/drive/folders/1_fBYbDO3XSdRt3DSbEBe41r5l9YpIGWF?usp=sharing)下载预处理好的数据，并保存到 `data/modelnet40_normal_resampled/` 目录下。
* 若要在 ModelNet10 上进行训练，可使用 `--num_category 10`。
```shell
# ModelNet40
## 在 ./models 中选择不同的模型

## 例如，无法线特征的 pointnet2_ssg
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
python test_classification.py --log_dir pointnet2_cls_ssg

## 例如，有法线特征的 pointnet2_ssg
python train_classification.py --model pointnet2_cls_ssg --use_normals --log_dir pointnet2_cls_ssg_normal
python test_classification.py --use_normals --log_dir pointnet2_cls_ssg_normal

## 例如，使用均匀采样的 pointnet2_ssg
python train_classification.py --model pointnet2_cls_ssg --use_uniform_sample --log_dir pointnet2_cls_ssg_fps
python test_classification.py --use_uniform_sample --log_dir pointnet2_cls_ssg_fps

# ModelNet10
## 设置与 ModelNet40 类似，只需使用 --num_category 10

## 例如，无法线特征的 pointnet2_ssg
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg --num_category 10
python test_classification.py --log_dir pointnet2_cls_ssg --num_category 10
```

### 性能表现
| 模型 | 准确率 |
|--|--|
| PointNet（官方） | 89.2 |
| PointNet2（官方） | 91.9 |
| PointNet（无法线的 PyTorch 实现） | 90.6 |
| PointNet（有法线的 PyTorch 实现） | 91.4 |
| PointNet2_SSG（无法线的 PyTorch 实现） | 92.2 |
| PointNet2_SSG（有法线的 PyTorch 实现） | 92.4 |
| PointNet2_MSG（有法线的 PyTorch 实现） | **92.8** |

## 部件分割（ShapeNet）
### 数据准备
从[此处](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)下载对齐后的 **ShapeNet** 数据集，并保存到 `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/` 目录下。
### 运行
```
## 在 ./models 中查看模型
## 例如，pointnet2_msg
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
python test_partseg.py --normal --log_dir pointnet2_part_seg_msg
```
### 性能表现
| 模型 | 实例平均交并比 | 类别平均交并比 |
|--|--|--|
| PointNet（官方） | 83.7 | 80.4 |
| PointNet2（官方） | 85.1 | 81.9 |
| PointNet（PyTorch 实现） | 84.3 | 81.1 |
| PointNet2_SSG（PyTorch 实现） | 84.9 | 81.8 |
| PointNet2_MSG（PyTorch 实现） | **85.4** | **82.5** |

## 语义分割（S3DIS）
### 数据准备
从[此处](http://buildingparser.stanford.edu/dataset.html)下载 3D 室内解析数据集（**S3DIS**），并保存到 `data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/` 目录下。
```
cd data_utils
python collect_indoor3d_data.py
```
处理后的数据将保存到 `data/stanford_indoor3d/` 目录下。
### 运行
```
## 在 ./models 中查看模型
## 例如，pointnet2_ssg
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5 --visual
```
可视化结果将保存到 `log/sem_seg/pointnet2_sem_seg/visual/` 目录下，你可以使用 [MeshLab](http://www.meshlab.net/) 来查看这些 .obj 文件。

### 性能表现
| 模型 | 总体准确率 | 类别平均交并比 | 检查点 |
|--|--|--|--|
| PointNet（PyTorch 实现） | 78.9 | 43.7 | [40.7MB](log/sem_seg/pointnet_sem_seg) |
| PointNet2_ssg（PyTorch 实现） | **83.0** | **53.5** | [11.2MB](log/sem_seg/pointnet2_sem_seg) |

## 可视化
### 使用 show3d_balls.py
```
## 编译用于可视化的 C++ 代码
cd visualizer
bash build.sh 
## 运行一个示例
python show3d_balls.py
```
![](/visualizer/pic.png)
### 使用 MeshLab
![](/visualizer/pic2.png)

## 参考来源
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)

## 引用说明
如果你在研究中发现此仓库很有用，请考虑引用它以及我们的其他成果：
```
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
```
```
@InProceedings{yan2020pointasnl,
  title={PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling},
  author={Yan, Xu and Zheng, Chaoda and Li, Zhen and Wang, Sheng and Cui, Shuguang},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
```
@InProceedings{yan2021sparse,
  title={Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion},
  author={Yan, Xu and Gao, Jiantao and Li, Jie and Zhang, Ruimao, and Li, Zhen and Huang, Rui and Cui, Shuguang},
  journal={AAAI Conference on Artificial Intelligence ({AAAI})},
  year={2021}
}
```
```
@InProceedings{yan20222dpass,
      title={2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds}, 
      author={Xu Yan and Jiantao Gao and Chaoda Zheng and Chao Zheng and Ruimao Zhang and Shuguang Cui and Zhen Li},
      year={2022},
      journal={ECCV}
}
```
## 使用此代码库的部分项目
* [PointConv: Deep Convolutional Networks on 3D Point Clouds, CVPR'19](https://github.com/Young98CN/pointconv_pytorch)
* [On Isometry Robustness of Deep 3D Point Cloud Models under Adversarial Attacks, CVPR'20](https://github.com/skywalker6174/3d-isometry-robust)
* [Label-Efficient Learning on Point Clouds using Approximate Convex Decompositions, ECCV'20](https://github.com/matheusgadelha/PointCloudLearningACD)
* [PCT: Point Cloud Transformer](https://github.com/MenghaoGuo/PCT)
* [PSNet: Fast Data Structuring for Hierarchical Deep Learning on Point Cloud](https://github.com/lly007/PointStructuringNet)
* [Stratified Transformer for 3D Point Cloud Segmentation, CVPR'22](https://github.com/dvlab-research/stratified-transformer)