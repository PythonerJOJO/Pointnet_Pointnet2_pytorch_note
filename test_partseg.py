"""
部件分割测试
Date: Nov 2019
cmd
python test_partseg.py --normal --log_dir pointnet2_part_seg_msg
"""

import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

# 获取当前脚本所在的目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
# 将模型目录添加到系统路径中，以便可以导入模型模块
sys.path.append(os.path.join(ROOT_DIR, "models"))

# 定义每个类别的部件标签
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

# 构建部件标签到类别名称的映射
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """
    对输入的标签张量进行one-hot编码
    :param y: 输入的标签张量
    :param num_classes: 类别总数
    :return: one-hot编码后的张量
    """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy()]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser("PointNet")
    parser.add_argument(
        "--batch_size", type=int, default=24, help="batch size in testing"
    )
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")
    parser.add_argument("--num_point", type=int, default=2048, help="point Number")
    parser.add_argument("--log_dir", type=str, required=True, help="experiment root")
    parser.add_argument(
        "--normal", action="store_true", default=False, help="use normals"
    )
    parser.add_argument(
        "--num_votes",
        type=int,
        default=3,
        help="aggregate segmentation scores with voting",
    )
    return parser.parse_args()


def main(args):
    def log_string(str):
        """
        记录日志并打印到控制台
        :param str: 要记录和打印的字符串
        """
        logger.info(str)
        print(str)

    """设置GPU环境变量"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # 实验目录路径
    experiment_dir = "log/part_seg/" + args.log_dir

    """配置日志系统"""
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler("%s/eval.txt" % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string("PARAMETER ...")
    log_string(args)

    # 数据集根目录
    root = "data/shapenetcore_partanno_segmentation_benchmark_v0_normal/"

    # 加载测试数据集
    TEST_DATASET = PartNormalDataset(
        root=root, npoints=args.num_point, split="test", normal_channel=args.normal
    )
    # 创建测试数据加载器
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    # 类别数量
    num_classes = 16
    # 部件数量
    num_part = 50

    """加载模型"""
    # 获取模型名称
    model_name = os.listdir(experiment_dir + "/logs")[0].split(".")[0]
    # 动态导入模型模块
    MODEL = importlib.import_module(model_name)
    # 初始化模型
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    # 加载预训练模型的权重
    checkpoint = torch.load(str(experiment_dir) + "/checkpoints/best_model.pth")
    classifier.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        # 初始化测试指标字典
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        # 每个部件类别的总样本数
        total_seen_class = [0 for _ in range(num_part)]
        # 每个部件类别的正确预测样本数
        total_correct_class = [0 for _ in range(num_part)]
        # 每个类别的IoU列表
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        # 重新构建部件标签到类别名称的映射
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        # 将模型设置为评估模式
        classifier = classifier.eval()
        for batch_id, (points, label, target) in tqdm(
            enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9
        ):
            # 获取当前批次的大小和点数
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            # 将数据移动到GPU上
            points, label, target = (
                points.float().cuda(),
                label.long().cuda(),
                target.long().cuda(),
            )
            # 调整点云数据的维度
            points = points.transpose(2, 1)
            # 初始化投票池
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            # 进行多次投票
            for _ in range(args.num_votes):
                # 前向传播，获取分割预测结果
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                # 累加投票结果
                vote_pool += seg_pred

            # 计算平均投票结果
            seg_pred = vote_pool / args.num_votes
            # 将预测结果移动到CPU上并转换为numpy数组
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            # 将目标标签移动到CPU上并转换为numpy数组
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                # 获取当前样本的类别
                cat = seg_label_to_cat[target[i, 0]]
                # 获取当前样本的预测logits
                logits = cur_pred_val_logits[i, :, :]
                # 获取当前样本的最终预测结果
                cur_pred_val[i, :] = (
                    np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                )

            # 计算当前批次的正确预测数量
            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += cur_batch_size * NUM_POINT

            # 统计每个部件类别的总样本数和正确预测样本数
            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += np.sum((cur_pred_val == l) & (target == l))

            # 计算每个样本的部件IoU
            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                        np.sum(segp == l) == 0
                    ):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum(
                            (segl == l) & (segp == l)
                        ) / float(np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        # 汇总所有形状的IoU
        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        # 计算平均形状IoU
        mean_shape_ious = np.mean(list(shape_ious.values()))
        # 计算整体准确率
        test_metrics["accuracy"] = total_correct / float(total_seen)
        # 计算每个部件类别的平均准确率
        test_metrics["class_avg_accuracy"] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
        )
        # 打印每个类别的平均IoU
        for cat in sorted(shape_ious.keys()):
            log_string(
                "eval mIoU of %s %f" % (cat + " " * (14 - len(cat)), shape_ious[cat])
            )
        # 计算所有类别的平均IoU
        test_metrics["class_avg_iou"] = mean_shape_ious
        # 计算所有实例的平均IoU
        test_metrics["inctance_avg_iou"] = np.mean(all_shape_ious)

    # 打印最终的测试指标
    log_string("Accuracy is: %.5f" % test_metrics["accuracy"])
    log_string("Class avg accuracy is: %.5f" % test_metrics["class_avg_accuracy"])
    log_string("Class avg mIOU is: %.5f" % test_metrics["class_avg_iou"])
    log_string("Inctance avg mIOU is: %.5f" % test_metrics["inctance_avg_iou"])


if __name__ == "__main__":
    args = parse_args()
    main(args)
