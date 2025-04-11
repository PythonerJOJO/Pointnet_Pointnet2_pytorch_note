"""
分类测试
日期：2019年11月
cmd：
python test_classification.py --use_normals --log_dir pointnet2_cls_msg

"""

from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))


def parse_args():
    """参数配置"""
    parser = argparse.ArgumentParser("测试配置")
    parser.add_argument(
        "--use_cpu", action="store_true", default=False, help="使用CPU模式"
    )
    parser.add_argument("--gpu", type=str, default="0", help="指定GPU设备")
    parser.add_argument("--batch_size", type=int, default=24, help="训练时的批量大小")
    parser.add_argument(
        "--num_category",
        default=40,
        type=int,
        choices=[10, 40],
        help="在ModelNet10/40上训练",
    )
    parser.add_argument("--num_point", type=int, default=1024, help="点云点数")
    parser.add_argument("--log_dir", type=str, required=True, help="实验日志根目录")
    parser.add_argument(
        "--use_normals", action="store_true", default=False, help="使用法向量特征"
    )
    parser.add_argument(
        "--use_uniform_sample", action="store_true", default=False, help="使用均匀采样"
    )
    parser.add_argument("--num_votes", type=int, default=3, help="通过投票聚合分类得分")
    return parser.parse_args()


def test(model, loader, num_class=40, vote_num=1):
    """测试分类模型
    参数:
        model: 分类模型
        loader: 数据加载器
        num_class: 类别数，默认40
        vote_num: 投票次数，默认1
    返回:
        instance_acc: 实例准确率
        class_acc: 类别平均准确率
    """
    mean_correct = []  # 存储每个批次的正确预测率
    classifier = model.eval()  # 设置为评估模式
    class_acc = np.zeros(
        (num_class, 3)
    )  # 类别准确率统计矩阵 [类别, [正确数, 样本数, 准确率]]

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()  # 数据转移到GPU

        points = points.transpose(2, 1)  # 调整维度为(B, C, N)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()  # 初始化投票池

        for _ in range(vote_num):
            pred, _ = classifier(points)  # 模型预测
            vote_pool += pred  # 累计投票
        pred = vote_pool / vote_num  # 平均投票结果
        pred_choice = pred.data.max(1)[1]  # 获取最大得分的类别索引

        # 按类别统计准确率
        for cat in np.unique(target.cpu()):
            # 计算当前类别的正确预测数
            classacc = (
                pred_choice[target == cat]
                .eq(target[target == cat].long().data)
                .cpu()
                .sum()
            )
            class_acc[cat, 0] += classacc.item() / float(
                points[target == cat].size()[0]
            )  # 累加类别准确率
            class_acc[cat, 1] += 1  # 记录类别样本数
        # 计算实例准确率
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    # 计算类别平均准确率
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)  # 计算实例平均准确率
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    """超参数配置"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 设置可见GPU设备

    """创建目录"""
    experiment_dir = "log/classification/" + args.log_dir  # 实验日志目录

    """日志配置"""
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler("%s/eval.txt" % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string("参数配置...")
    log_string(args)  # 打印参数配置

    """数据加载"""
    log_string("加载数据集...")
    data_path = "data/modelnet40_normal_resampled/"  # ModelNet数据集路径

    # 初始化测试数据集
    test_dataset = ModelNetDataLoader(
        root=data_path, args=args, split="test", process_data=False
    )
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10
    )  # 测试数据加载器

    """模型加载"""
    num_class = args.num_category
    # 自动获取模型名称（从日志文件名解析）
    model_name = os.listdir(experiment_dir + "/logs")[0].split(".")[0]
    model = importlib.import_module(model_name)  # 动态导入模型模块

    # 初始化分类器
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()  # 模型转移到GPU

    # 加载最佳模型权重
    checkpoint = torch.load(str(experiment_dir) + "/checkpoints/best_model.pth")
    classifier.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():  # 关闭梯度计算
        # 执行测试
        instance_acc, class_acc = test(
            classifier.eval(),
            testDataLoader,
            vote_num=args.num_votes,
            num_class=num_class,
        )
        log_string("测试实例准确率: %f, 类别平均准确率: %f" % (instance_acc, class_acc))


if __name__ == "__main__":
    args = parse_args()
    main(args)
