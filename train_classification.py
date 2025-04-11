"""
分类训练
日期：2019年11月
命令示例：
python train_classification.py --model pointnet2_cls_msg --use_normals --log_dir pointnent2_cls_msg_normal --batch_size 16
"""

import os
import sys
import torch
import numpy as np
import datetime
import logging
import provider
import importlib
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))


def parse_args():
    """参数配置"""
    parser = argparse.ArgumentParser("训练配置")
    parser.add_argument(
        "--use_cpu", action="store_true", default=False, help="使用CPU模式"
    )
    parser.add_argument("--gpu", type=str, default="0", help='指定GPU设备（如"0,1"）')
    parser.add_argument("--batch_size", type=int, default=24, help="训练时的批量大小")
    parser.add_argument(
        "--model", default="pointnet_cls", help="模型名称 [默认: pointnet_cls]"
    )
    parser.add_argument(
        "--num_category",
        default=40,
        type=int,
        choices=[10, 40],
        help="在ModelNet10/40数据集上训练",
    )
    parser.add_argument("--epoch", default=200, type=int, help="训练的epoch数量")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="训练学习率")
    parser.add_argument(
        "--num_point", type=int, default=1024, help="每批点云包含的点数"
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="优化器类型（Adam/SGD）"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="实验日志根目录（若未指定则自动生成时间戳目录）",
    )
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="权重衰减率")
    parser.add_argument(
        "--use_normals", action="store_true", default=False, help="是否使用法向量特征"
    )
    parser.add_argument(
        "--process_data",
        action="store_true",
        default=False,
        help="是否离线预处理数据（加速加载）",
    )
    parser.add_argument(
        "--use_uniform_sample",
        action="store_true",
        default=False,
        help="是否使用均匀采样策略",
    )
    return parser.parse_args()


def inplace_relu(m):
    """将ReLU层设置为inplace模式以节省内存
    参数:
        m: 神经网络模块
    """
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True  # 启用inplace操作，直接在输入张量上执行ReLU以减少内存占用


def test(model, loader, num_class=40):
    """模型测试函数
    参数:
        model: 待测试的分类模型
        loader: 测试数据加载器
        num_class: 类别总数，默认40（对应ModelNet40）
    返回:
        instance_acc: 实例平均准确率（所有样本的平均正确分类率）
        class_acc: 类别平均准确率（每个类别的平均正确分类率）
    """
    mean_correct = []  # 存储每个批次的实例准确率
    class_acc = np.zeros(
        (num_class, 3)
    )  # 类别准确率统计矩阵 [类别索引, [正确比例累加, 样本数累加, 最终准确率]]
    classifier = model.eval()  # 设置模型为评估模式（关闭dropout/batchnorm等）

    for j, (points, target) in tqdm(
        enumerate(loader), total=len(loader), desc="测试进度"
    ):
        # 数据设备转移
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)  # 调整维度为(B, C, N)以适应模型输入
        pred, _ = classifier(points)  # 模型前向传播，获取预测结果
        pred_choice = pred.data.max(1)[
            1
        ]  # 获取每个样本的预测类别（取logits最大值的索引）

        # 按类别统计准确率
        for cat in np.unique(target.cpu()):  # 遍历当前批次中的所有类别
            # 计算当前类别正确预测的比例并累加到class_acc
            correct_mask = pred_choice[target == cat].eq(
                target[target == cat].long().data
            )
            classacc = correct_mask.cpu().sum().item() / float(
                points[target == cat].size()[0]
            )
            class_acc[cat, 0] += classacc  # 累加类别正确比例
            class_acc[cat, 1] += 1  # 记录该类别样本数

        # 计算实例准确率（所有类别混合后的平均准确率）
        correct = pred_choice.eq(target.long().data).cpu().sum().item()
        mean_correct.append(correct / float(points.size()[0]))

    # 计算最终类别平均准确率
    class_acc[:, 2] = (
        class_acc[:, 0] / class_acc[:, 1]
    )  # 每个类别准确率 = 正确比例总和 / 样本数总和
    class_acc = np.mean(class_acc[:, 2])  # 所有类别准确率的平均值
    instance_acc = np.mean(mean_correct)  # 所有批次实例准确率的平均值
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        """日志输出函数（同时输出到控制台和日志文件）"""
        logger.info(str)
        print(str)

    """超参数配置"""
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        args.gpu
    )  # 设置可见的GPU设备（支持多卡，如"0,1"）

    """创建实验目录"""
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")  # 生成时间戳
    exp_root = Path("./log/classification/")
    exp_root.mkdir(exist_ok=True, parents=True)  # 创建日志根目录

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
    logger = logging.getLogger("ModelTraining")
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
    log_string("加载数据集...")
    # data_path = 'data/modelnet40_normal_resampled/'  # ModelNet数据集路径
    data_path = "/home/dai/study_test/ai/data/modelnet40_normal_resampled/"  # ModelNet数据集路径

    # 初始化数据加载器
    train_dataset = ModelNetDataLoader(
        root=data_path, args=args, split="train", process_data=args.process_data
    )
    test_dataset = ModelNetDataLoader(
        root=data_path, args=args, split="test", process_data=args.process_data
    )
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
        drop_last=True,
    )  # 训练数据加载器（打乱顺序，丢弃不完整批次）
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10
    )  # 测试数据加载器（顺序加载）

    """模型加载与初始化"""
    num_class = args.num_category
    model_module = importlib.import_module(
        args.model
    )  # 动态导入模型模块（如pointnet_cls）

    # 保存代码副本到实验目录（方便复现）
    shutil.copy(f"./models/{args.model}.py", exp_dir)
    shutil.copy("models/pointnet2_utils.py", exp_dir)
    shutil.copy("./train_classification.py", exp_dir)

    # 初始化模型和损失函数
    classifier = model_module.get_model(num_class, normal_channel=args.use_normals)
    criterion = model_module.get_loss()  # 获取模型对应的损失函数（如带正则项的交叉熵）
    classifier.apply(inplace_relu)  # 应用inplace_relu函数优化内存

    # 设备转移
    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    """加载预训练模型（如果存在）"""
    best_instance_acc = 0.0
    best_class_acc = 0.0
    start_epoch = 0
    try:
        checkpoint = torch.load(checkpoints_dir / "best_model.pth")
        start_epoch = checkpoint["epoch"]
        classifier.load_state_dict(checkpoint["model_state_dict"])
        best_instance_acc = checkpoint["instance_acc"]
        best_class_acc = checkpoint["class_acc"]
        log_string("加载预训练模型成功")
    except FileNotFoundError:
        log_string("未找到预训练模型，从0开始训练...")

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
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.7
    )  # 学习率衰减策略

    """训练主循环"""
    logger.info("开始训练...")
    global_epoch = 0
    global_step = 0
    for epoch in range(start_epoch, args.epoch):
        log_string(f"第 {global_epoch + 1} 个epoch（共 {args.epoch} 个）:")
        mean_correct = []
        classifier.train()  # 设置模型为训练模式（启用dropout/batchnorm）

        scheduler.step()  # 更新学习率
        for batch_id, (points, target) in tqdm(
            enumerate(trainDataLoader), total=len(trainDataLoader), desc="训练进度"
        ):
            optimizer.zero_grad()  # 梯度清零

            # 数据增强（转换为numpy数组以使用provider模块函数）
            points = points.data.numpy()
            points = provider.random_point_dropout(points)  # 随机丢弃部分点
            points[:, :, 0:3] = provider.random_scale_point_cloud(
                points[:, :, 0:3]
            )  # 随机缩放坐标
            points[:, :, 0:3] = provider.shift_point_cloud(
                points[:, :, 0:3]
            )  # 随机平移坐标
            points = torch.Tensor(points)
            points = points.transpose(2, 1)  # 调整维度为(B, C, N)

            # 数据设备转移
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            # 前向传播与损失计算
            pred, trans_feat = classifier(
                points
            )  # 获取预测结果和特征变换矩阵（用于正则化）
            loss = criterion(
                pred, target.long(), trans_feat
            )  # 计算损失（含特征对齐正则项）
            pred_choice = pred.data.max(1)[1]  # 获取预测类别

            # 计算当前批次实例准确率
            correct = pred_choice.eq(target.long().data).cpu().sum().item()
            mean_correct.append(correct / float(points.size()[0]))

            # 反向传播与优化
            loss.backward()
            optimizer.step()
            global_step += 1

        # 计算训练集实例准确率
        train_instance_acc = np.mean(mean_correct)
        log_string(f"训练实例准确率: {train_instance_acc:.6f}")

        # 测试集评估（关闭梯度计算）
        with torch.no_grad():
            instance_acc, class_acc = test(
                classifier.eval(), testDataLoader, num_class=num_class
            )

            # 更新最佳模型记录
            if instance_acc >= best_instance_acc:
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
            if class_acc >= best_class_acc:
                best_class_acc = class_acc

            log_string(
                f"测试实例准确率: {instance_acc:.6f}, 类别准确率: {class_acc:.6f}"
            )
            log_string(
                f"最佳实例准确率: {best_instance_acc:.6f}, 类别准确率: {best_class_acc:.6f}"
            )

            # 保存最佳模型（基于实例准确率）
            if instance_acc >= best_instance_acc:
                save_path = checkpoints_dir / "best_model.pth"
                log_string(f"保存模型到 {save_path}")
                state = {
                    "epoch": best_epoch,
                    "instance_acc": instance_acc,
                    "class_acc": class_acc,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, save_path)

        global_epoch += 1  # 全局epoch计数

    logger.info("训练结束...")


if __name__ == "__main__":
    args = parse_args()
    main(args)
