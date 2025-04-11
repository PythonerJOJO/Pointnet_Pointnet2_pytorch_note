"""
部件分割训练
日期：2019年11月
cmd:
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
"""

import argparse
import os
from sympy import false
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))

# 定义每个类别的部件标签（类别名称: [部件标签列表]）
seg_classes = {
    "Earphone": [16, 17, 18],
    "Motorbike": [30, 31, 32, 33, 34, 35],
    "Rocket": [41, 42, 43],
    "Car": [8, 9, 10, 11],
    "Laptop": [28, 29],
    "Cap": [6, 7],
    "Skateboard": [44, 45, 46],
    "Mug": [36, 37],
    "Guitar": [19, 20, 21],
    "Bag": [4, 5],
    "Lamp": [24, 25, 26, 27],
    "Table": [47, 48, 49],
    "Airplane": [0, 1, 2, 3],
    "Pistol": [38, 39, 40],
    "Chair": [12, 13, 14, 15],
    "Knife": [22, 23],
}
# 部件标签到类别的映射（{部件标签: 类别名称}）
seg_label_to_cat = {}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    """将ReLU层设置为inplace模式以节省内存
    参数:
        m: 神经网络模块
    """
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True  # 启用inplace操作，直接在输入张量上执行ReLU以减少内存占用


def to_categorical(y, num_classes):
    """对张量进行one-hot编码
    参数:
        y: 输入标签张量（形状为[B]）
        num_classes: 总类别数
    返回:
        one-hot编码后的张量（形状为[B, num_classes]）
    """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy()]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser("部件分割训练配置")
    parser.add_argument(
        "--model", type=str, default="pointnet_part_seg", help="模型名称"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="训练时的批量大小")
    parser.add_argument("--epoch", default=251, type=int, help="训练的总epoch数")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="初始学习率")
    parser.add_argument("--gpu", type=str, default="0", help='指定GPU设备（如"0,1"）')
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="优化器类型（Adam或SGD）"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="日志保存目录（若未指定则自动生成时间戳目录）",
    )
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="权重衰减率")
    parser.add_argument("--npoint", type=int, default=2048, help="每批点云包含的点数")
    parser.add_argument(
        "--normal", action="store_true", default=False, help="是否使用法向量特征"
    )
    parser.add_argument("--step_size", type=int, default=20, help="学习率衰减的步长")
    parser.add_argument("--lr_decay", type=float, default=0.5, help="学习率衰减率")
    return parser.parse_args()


def main(args):
    def log_string(str):
        """日志输出函数（同时输出到控制台和日志文件）"""
        logger.info(str)
        print(str)

    """超参数配置"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 设置可见的GPU设备

    """创建实验目录"""
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")  # 生成时间戳
    exp_root = Path("./log/part_seg/")
    exp_root.mkdir(exist_ok=True, parents=True)  # 创建部件分割日志根目录

    # 确定实验目录：使用用户指定目录或自动生成时间戳目录
    if args.log_dir is None:
        exp_dir = exp_root / timestr
    else:
        exp_dir = exp_root / args.log_dir
    exp_dir.mkdir(exist_ok=True)

    # 创建子目录
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    """日志系统配置"""
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 文件日志处理器（保存训练日志）
    file_handler = logging.FileHandler(log_dir / f"{args.model}.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_string("参数配置信息...")
    log_string(args)  # 打印完整的参数配置

    """数据加载配置"""
    # root = "data/shapenetcore_partanno_segmentation_benchmark_v0_normal/"  # ShapeNet部件分割数据集路径
    root = "d:/0_study_test/AI/point_cloud/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/"  # ShapeNet部件分割数据集路径

    # 初始化数据加载器
    TRAIN_DATASET = PartNormalDataset(
        root=root, npoints=args.npoint, split="trainval", normal_channel=args.normal
    )
    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
        drop_last=True,
    )  # 训练数据加载器（打乱顺序，丢弃不完整批次）
    TEST_DATASET = PartNormalDataset(
        root=root, npoints=args.npoint, split="test", normal_channel=args.normal
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10
    )  # 测试数据加载器（顺序加载）

    log_string(f"训练数据数量: {len(TRAIN_DATASET)}")
    log_string(f"测试数据数量: {len(TEST_DATASET)}")

    num_classes = 16  # 物体类别数（ShapeNet部件分割共16个物体类别）
    num_part = 50  # 部件标签总数（所有类别共有50种部件）

    """模型加载与初始化"""
    MODEL = importlib.import_module(
        args.model
    )  # 动态导入模型模块（如pointnet_part_seg）

    # 保存代码副本到实验目录（方便复现）
    shutil.copy(f"models/{args.model}.py", exp_dir)
    shutil.copy("models/pointnet2_utils.py", exp_dir)

    # 初始化模型和损失函数
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()  # 获取模型对应的损失函数（含特征对齐正则项）
    classifier.apply(inplace_relu)  # 应用inplace_relu函数优化内存

    def weights_init(m):
        """初始化模型权重（Xavier正态分布初始化）"""
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("Linear") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    """加载预训练模型（如果存在）"""
    best_acc = 0.0
    best_class_avg_iou = 0.0
    best_instance_avg_iou = 0.0
    start_epoch = 0
    try:
        # TODO: torch 2.6 改了load方法
        # checkpoint = torch.load(checkpoints_dir / "best_model.pth")
        checkpoint = torch.load(checkpoints_dir / "best_model.pth", weights_only=False)
        start_epoch = checkpoint["epoch"]
        classifier.load_state_dict(checkpoint["model_state_dict"])
        log_string("加载预训练模型成功")
    except FileNotFoundError:
        log_string("未找到预训练模型，从0开始训练...")
        classifier = classifier.apply(weights_init)  # 随机初始化权重

    """优化器配置"""
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(), lr=args.learning_rate, momentum=0.9
        )

    """学习率与BN动量衰减配置"""
    LEARNING_RATE_CLIP = 1e-5  # 学习率下限
    MOMENTUM_ORIGINAL = 0.1  # 初始BN动量
    MOMENTUM_DECCAY = 0.5  # BN动量衰减率
    MOMENTUM_DECAY_STEP = args.step_size  # BN动量衰减步长

    """训练主循环"""
    # bn_momentum_adjust = 0
    for epoch in range(start_epoch, args.epoch):
        mean_correct = []  # 存储每个批次的实例准确率

        log_string(f"第 {epoch + 1} 个epoch（共 {args.epoch} 个）:")
        """调整学习率和BN动量"""
        lr = max(
            args.learning_rate * (args.lr_decay ** (epoch // args.step_size)),
            LEARNING_RATE_CLIP,
        )
        log_string(f"学习率调整为: {lr:.8f}")
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr  # 更新优化器学习率

        momentum = MOMENTUM_ORIGINAL * (
            MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECAY_STEP)
        )
        momentum = max(momentum, 0.01)  # 确保动量不低于0.01
        log_string(f"BN动量调整为: {momentum:.4f}")
        classifier = classifier.apply(
            lambda x: bn_momentum_adjust(x, momentum)
        )  # 更新BN层动量
        classifier.train()  # 设置模型为训练模式

        """训练一个epoch"""
        for i, (points, label, target) in tqdm(
            enumerate(trainDataLoader), total=len(trainDataLoader), desc="训练进度"
        ):
            optimizer.zero_grad()  # 梯度清零

            # 数据增强（转换为numpy数组以使用provider模块函数）
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(
                points[:, :, 0:3]
            )  # 随机缩放坐标
            points[:, :, 0:3] = provider.shift_point_cloud(
                points[:, :, 0:3]
            )  # 随机平移坐标
            points = torch.Tensor(points)

            # 数据设备转移（GPU/CPU）
            points, label, target = (
                points.float().cuda(),
                label.long().cuda(),
                target.long().cuda(),
            )
            points = points.transpose(2, 1)  # 调整维度为(B, C, N)

            # 前向传播获取分割预测和特征变换矩阵
            seg_pred, trans_feat = classifier(
                points, to_categorical(label, num_classes)
            )
            seg_pred = seg_pred.contiguous().view(-1, num_part)  # 展平为[B*N, num_part]
            target_flat = target.view(-1)  # 展平为[B*N]
            pred_choice = seg_pred.data.max(1)[1]  # 获取每个点的预测部件标签

            # 计算当前批次准确率
            correct = pred_choice.eq(target_flat.data).cpu().sum()
            mean_correct.append(
                correct.item() / (args.batch_size * args.npoint)
            )  # 实例准确率

            # 计算损失并反向传播
            loss = criterion(seg_pred, target_flat, trans_feat)
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        log_string(f"训练实例准确率: {train_instance_acc:.5f}")

        """测试集评估（关闭梯度计算）"""
        with torch.no_grad():
            test_metrics = {}
            total_correct = 0  # 总正确预测点数
            total_seen = 0  # 总点数
            total_seen_class = [0] * num_part  # 每个部件标签的总出现次数
            total_correct_class = [0] * num_part  # 每个部件标签的正确预测次数
            shape_ious = {
                cat: [] for cat in seg_classes.keys()
            }  # 存储每个类别的形状IoU

            classifier.eval()  # 设置模型为评估模式

            for batch_id, (points, label, target) in tqdm(
                enumerate(testDataLoader), total=len(testDataLoader), desc="测试进度"
            ):
                batch_size, num_point, _ = points.size()
                points, label, target = (
                    points.float().cuda(),
                    label.long().cuda(),
                    target.long().cuda(),
                )
                points = points.transpose(2, 1)  # 调整维度为(B, C, N)

                # 前向传播获取分割预测
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                seg_pred_logits = seg_pred.cpu().data.numpy()  # 获取logits
                seg_pred_labels = np.zeros(
                    (batch_size, num_point), dtype=np.int32
                )  # 初始化预测标签

                for i in range(batch_size):
                    cat = seg_label_to_cat[target[i, 0].item()]  # 获取当前形状的类别
                    class_seg_classes = seg_classes[cat]  # 获取该类别对应的部件标签列表
                    logits = seg_pred_logits[
                        i, :, class_seg_classes
                    ]  # 提取该类别相关的logits
                    seg_pred_labels[i, :] = (
                        np.argmax(logits, axis=1) + class_seg_classes[0]
                    )  # 映射回全局部件标签

                # 计算全局准确率
                correct = np.sum(seg_pred_labels == target.cpu().data.numpy())
                total_correct += correct
                total_seen += batch_size * num_point

                # 按部件标签统计准确率
                for l in range(num_part):
                    total_seen_class[l] += np.sum(target.cpu().data.numpy() == l)
                    total_correct_class[l] += np.sum(
                        (seg_pred_labels == l) & (target.cpu().data.numpy() == l)
                    )

                # 计算每个形状的部件IoU
                for i in range(batch_size):
                    pred = seg_pred_labels[i, :]
                    gt = target[i, :].cpu().data.numpy()
                    cat = seg_label_to_cat[gt[0]]  # 获取形状类别
                    class_parts = seg_classes[cat]  # 该类别对应的部件列表
                    part_ious = []
                    for idx, l in enumerate(class_parts):
                        # 计算单个部件的IoU：交集/并集（处理空集情况）
                        intersect = np.sum((pred == l) & (gt == l))
                        union = np.sum((pred == l) | (gt == l))
                        if union == 0:
                            part_ious.append(1.0)  # 无该部件时视为IoU=1
                        else:
                            part_ious.append(intersect / union)
                    shape_ious[cat].append(
                        np.mean(part_ious)
                    )  # 存储该形状的平均部件IoU

            # 计算整体指标
            test_metrics["accuracy"] = total_correct / total_seen  # 整体准确率
            test_metrics["class_avg_accuracy"] = (
                np.mean(  # 类别平均准确率（每个部件标签的准确率平均）
                    np.array(total_correct_class, dtype=np.float)
                    / np.array(total_seen_class, dtype=np.float)
                )
            )
            all_shape_ious = [
                iou for cat_ious in shape_ious.values() for iou in cat_ious
            ]
            test_metrics["class_avg_iou"] = np.mean(
                list(shape_ious.values())
            )  # 类别平均IoU（每个物体类别的平均IoU）
            test_metrics["instance_avg_iou"] = np.mean(
                all_shape_ious
            )  # 实例平均IoU（所有形状的平均IoU）

            # 打印每个类别的IoU
            for cat in sorted(shape_ious.keys()):
                log_string(
                    f"{cat.ljust(14)} mIoU: {shape_ious[cat]:.4f}"
                )  # ljust用于对齐输出

        # 记录测试结果
        log_string(
            f"Epoch {epoch+1} 测试结果 | "
            f'准确率: {test_metrics["accuracy"]:.5f} | '
            f'类别平均IoU: {test_metrics["class_avg_iou"]:.5f} | '
            f'实例平均IoU: {test_metrics["instance_avg_iou"]:.5f}'
        )

        """保存最佳模型（基于实例平均IoU）"""
        if test_metrics["instance_avg_iou"] >= best_instance_avg_iou:
            save_path = checkpoints_dir / "best_model.pth"
            log_string(
                f'保存模型到 {save_path}（当前实例平均IoU: {test_metrics["instance_avg_iou"]:.5f}）'
            )
            state = {
                "epoch": epoch,
                "train_acc": train_instance_acc,
                "test_acc": test_metrics["accuracy"],
                "class_avg_iou": test_metrics["class_avg_iou"],
                "instance_avg_iou": test_metrics["instance_avg_iou"],
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(state, save_path)

        # 更新最佳指标记录
        best_acc = max(best_acc, test_metrics["accuracy"])
        best_class_avg_iou = max(best_class_avg_iou, test_metrics["class_avg_iou"])
        best_instance_avg_iou = max(
            best_instance_avg_iou, test_metrics["instance_avg_iou"]
        )
        log_string(
            f"当前最佳 | 准确率: {best_acc:.5f} | 类别平均IoU: {best_class_avg_iou:.5f} | 实例平均IoU: {best_instance_avg_iou:.5f}"
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
