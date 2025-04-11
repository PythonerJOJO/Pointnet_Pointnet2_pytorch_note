
"""
场景分割训练
日期：2019年11月
"""

import argparse
import os
from data_utils.S3DISDataLoader import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# 定义场景分割的类别及其标签（共13类）
classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}  # 类别到标签的映射（如'floor': 1）
seg_label_to_cat = {i: cat for i, cat in enumerate(classes)}  # 标签到类别的映射（如1: 'floor'）

def inplace_relu(m):
    """将ReLU层设置为inplace模式以节省内存
    参数:
        m: 神经网络模块
    """
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True  # 启用inplace操作，直接在输入张量上执行ReLU以减少内存占用

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser('场景分割训练配置')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='模型名称 [默认: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='训练时的批量大小 [默认: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='训练的总epoch数 [默认: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='初始学习率 [默认: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='使用的GPU设备 [默认: GPU 0]（如"0,1"指定多卡）')
    parser.add_argument('--optimizer', type=str, default='Adam', help='优化器类型 [默认: Adam]（可选Adam或SGD）')
    parser.add_argument('--log_dir', type=str, default=None, help='日志保存目录 [默认: 自动生成时间戳目录]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='权重衰减率 [默认: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='每批点云包含的点数 [默认: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='学习率衰减步长 [默认: 每10个epoch衰减一次]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='学习率衰减率 [默认: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='测试使用的区域（1-6） [默认: 5]')
    return parser.parse_args()

def main(args):
    def log_string(str):
        """日志输出函数（同时输出到控制台和日志文件）"""
        logger.info(str)
        print(str)

    '''超参数配置'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 设置可见的GPU设备

    '''创建实验目录'''
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')  # 生成时间戳
    exp_root = Path('./log/sem_seg/')
    exp_root.mkdir(exist_ok=True, parents=True)  # 创建场景分割日志根目录

    # 确定实验目录：使用用户指定目录或自动生成时间戳目录
    if args.log_dir is None:
        exp_dir = exp_root / timestr
    else:
        exp_dir = exp_root / args.log_dir
    exp_dir.mkdir(exist_ok=True)

    # 创建子目录
    checkpoints_dir = exp_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    '''日志系统配置'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 文件日志处理器（保存训练日志）
    file_handler = logging.FileHandler(log_dir / f'{args.model}.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    log_string('参数配置信息...')
    log_string(args)  # 打印完整的参数配置

    '''数据加载配置'''
    root = 'data/stanford_indoor3d/'  # S3DIS数据集路径
    NUM_CLASSES = 13  # 场景类别数（与classes列表长度一致）
    NUM_POINT = args.npoint  # 每批点云点数
    BATCH_SIZE = args.batch_size  # 批量大小

    # 初始化数据加载器（训练集和测试集）
    print("开始加载训练数据...")
    TRAIN_DATASET = S3DISDataset(
        split='train', 
        data_root=root, 
        num_point=NUM_POINT, 
        test_area=args.test_area, 
        block_size=1.0, 
        sample_rate=1.0, 
        transform=None
    )
    print("开始加载测试数据...")
    TEST_DATASET = S3DISDataset(
        split='test', 
        data_root=root, 
        num_point=NUM_POINT, 
        test_area=args.test_area, 
        block_size=1.0, 
        sample_rate=1.0, 
        transform=None
    )

    # 配置数据加载器（训练集打乱顺序，测试集顺序加载）
    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=10, 
        pin_memory=True, 
        drop_last=True, 
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=10, 
        pin_memory=True, 
        drop_last=True
    )
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()  # 类别权重（处理类别不平衡）

    log_string(f"训练数据数量: {len(TRAIN_DATASET)}")
    log_string(f"测试数据数量: {len(TEST_DATASET)}")

    '''模型加载与初始化'''
    MODEL = importlib.import_module(args.model)  # 动态导入模型模块（如pointnet_sem_seg）
    
    # 保存代码副本到实验目录（方便复现）
    shutil.copy(f'models/{args.model}.py', exp_dir)
    shutil.copy('models/pointnet2_utils.py', exp_dir)

    # 初始化模型和损失函数
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()  # 获取带权重的交叉熵损失函数
    classifier.apply(inplace_relu)  # 应用inplace_relu优化内存

    def weights_init(m):
        """初始化模型权重（Xavier正态分布初始化）"""
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    '''加载预训练模型（如果存在）'''
    best_iou = 0.0
    start_epoch = 0
    try:
        checkpoint = torch.load(checkpoints_dir / 'best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('加载预训练模型成功')
    except FileNotFoundError:
        log_string('未找到预训练模型，从0开始训练...')
        classifier = classifier.apply(weights_init)  # 随机初始化权重

    '''优化器配置'''
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        """调整BN层的动量参数"""
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum  # 更新BN层的动量

    LEARNING_RATE_CLIP = 1e-5  # 学习率下限
    MOMENTUM_ORIGINAL = 0.1     # 初始BN动量
    MOMENTUM_DECCAY = 0.5      # BN动量衰减率
    MOMENTUM_DECAY_STEP = args.step_size  # BN动量衰减步长

    '''训练主循环'''
    for epoch in range(start_epoch, args.epoch):
        log_string(f'**** 第 {global_epoch + 1} 个epoch（共 {args.epoch} 个） ****')
        '''调整学习率和BN动量'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string(f'学习率调整为: {lr:.8f}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 更新优化器学习率
        
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECAY_STEP))
        momentum = max(momentum, 0.01)  # 确保动量不低于0.01
        log_string(f'BN动量调整为: {momentum:.4f}')
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))  # 更新BN层动量
        classifier.train()  # 设置模型为训练模式

        '''训练阶段'''
        total_correct = 0  # 总正确预测点数
        total_seen = 0     # 总处理点数
        loss_sum = 0.0     # 总损失
        num_batches = len(trainDataLoader)

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=num_batches, desc="训练进度"):
            optimizer.zero_grad()  # 梯度清零

            # 数据增强：绕Z轴随机旋转（场景分割常用增强方式）
            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            
            # 数据设备转移（GPU/CPU）
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)  # 调整维度为(B, C, N)

            # 前向传播获取分割预测和特征变换矩阵
            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)  # 展平为[B*N, NUM_CLASSES]

            # 计算损失（带类别权重的交叉熵）
            target_flat = target.view(-1)  # 展平为[B*N]
            loss = criterion(seg_pred, target_flat, trans_feat, weights)
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新

            # 计算当前批次准确率
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()  # 获取预测标签
            batch_label = target_flat.cpu().data.numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += BATCH_SIZE * NUM_POINT
            loss_sum += loss.item()

        log_string(f'训练平均损失: {loss_sum / num_batches:.4f}')
        log_string(f'训练准确率: {total_correct / total_seen:.5f}')

        '''每5个epoch保存模型（非最佳模型，用于中间结果）'''
        if epoch % 5 == 0:
            save_path = checkpoints_dir / 'model.pth'
            log_string(f'保存中间模型到 {save_path}')
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, save_path)

        '''测试阶段（关闭梯度计算）'''
        with torch.no_grad():
            total_correct = 0         # 总正确预测点数
            total_seen = 0            # 总点数
            loss_sum = 0.0            # 总损失
            total_seen_class = [0] * NUM_CLASSES  # 每个类别的总出现次数
            total_correct_class = [0] * NUM_CLASSES  # 每个类别的正确预测次数
            total_iou_deno_class = [0] * NUM_CLASSES  # 每个类别的IoU分母（并集大小）
            classifier.eval()  # 设置模型为评估模式

            log_string(f'---- 第 {global_epoch + 1} 个epoch 评估 ----')
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), desc="测试进度"):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)  # 调整维度为(B, C, N)

                # 前向传播获取分割预测
                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()  # 获取logits
                seg_pred_flat = seg_pred.contiguous().view(-1, NUM_CLASSES)  # 展平为[B*N, NUM_CLASSES]

                # 计算损失
                target_flat = target.view(-1)
                loss = criterion(seg_pred_flat, target_flat, trans_feat, weights)
                loss_sum += loss.item()

                # 计算准确率和IoU相关统计
                pred_label = np.argmax(pred_val, axis=2)  # 预测标签形状为[B, N]
                batch_label = target.cpu().data.numpy()  # 真实标签形状为[B, N]
                correct = np.sum(pred_label == batch_label)
                total_correct += correct
                total_seen += BATCH_SIZE * NUM_POINT

                # 按类别统计
                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum(batch_label == l)
                    total_correct_class[l] += np.sum((pred_label == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum((pred_label == l) | (batch_label == l))

            # 计算评估指标
            mIoU = np.mean([total_correct_class[l] / max(total_iou_deno_class[l], 1) for l in range(NUM_CLASSES)])  # 平均IoU
            class_avg_acc = np.mean([total_correct_class[l] / max(total_seen_class[l], 1) for l in range(NUM_CLASSES)])  # 类别平均准确率
            overall_acc = total_correct / total_seen  # 整体准确率

            log_string(f'评估平均损失: {loss_sum / len(testDataLoader):.4f}')
            log_string(f'评估点云平均准确率: {overall_acc:.5f}')
            log_string(f'评估类别平均IoU: {mIoU:.5f}')
            log_string(f'评估类别平均准确率: {class_avg_acc:.5f}')

            # 打印每个类别的IoU细节
            iou_detail = '------- 类别IoU详情 -------\n'
            for l in range(NUM_CLASSES):
                cat_name = seg_label_to_cat[l]
                iou = total_correct_class[l] / max(total_iou_deno_class[l], 1)
                iou_detail += f'{cat_name.ljust(14)} IoU: {iou:.3f}\n'  # ljust用于左对齐
            log_string(iou_detail)

            '''保存最佳模型（基于平均IoU）'''
            if mIoU >= best_iou:
                best_iou = mIoU
                save_path = checkpoints_dir / 'best_model.pth'
                log_string(f'保存最佳模型到 {save_path}（当前mIoU: {mIoU:.5f}）')
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, save_path)

            log_string(f'当前最佳平均IoU: {best_iou:.5f}')
        global_epoch += 1  # 全局epoch计数


if __name__ == '__main__':
    args = parse_args()
    main(args)