"""
场景分割测试
日期：2019年11月
"""
import argparse
import os
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''参数配置'''
    parser = argparse.ArgumentParser('模型测试')
    parser.add_argument('--batch_size', type=int, default=32, help='测试时的批量大小 [默认: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='指定GPU设备')
    parser.add_argument('--num_point', type=int, default=4096, help='点云点数 [默认: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='实验根目录')
    parser.add_argument('--visual', action='store_true', default=False, help='可视化结果 [默认: False]')
    parser.add_argument('--test_area', type=int, default=5, help='测试区域（选项：1-6） [默认: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='通过投票聚合分割得分 [默认: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    """向投票池添加预测标签投票
    参数:
        vote_label_pool: 投票池数组 (点云总数 x 类别数)
        point_idx: 点云索引数组 (BxN)
        pred_label: 预测标签数组 (BxN)
        weight: 权重数组 (BxN)
    返回:
        更新后的投票池
    """
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        """日志输出函数（同时输出到控制台和日志文件）"""
        logger.info(str)
        print(str)

    '''超参数配置'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''日志配置'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('参数配置...')
    log_string(args)

    NUM_CLASSES = 13
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    root = 'data/s3dis/stanford_indoor3d/'

    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT)
    log_string("测试数据数量：%d" % len(TEST_DATASET_WHOLE_SCENE))

    '''模型加载'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- 整体场景评估 ----')

        for batch_idx in range(num_batches):
            print(f"推理 [{batch_idx + 1}/{num_batches}] {scene_id[batch_idx]} ...")
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            if args.visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))
                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    batch_data[:, :, 3:6] /= 1.0  # 归一化法线特征

                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)  # 调整维度为(B, C, N)
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()  # 获取预测标签

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)  # 获取最终预测标签

            # 统计当前场景的指标
            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] = np.sum(whole_scene_label == l)
                total_correct_class_tmp[l] = np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] = np.sum((pred_label == l) | (whole_scene_label == l))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = total_correct_class_tmp / (total_iou_deno_class_tmp.astype(np.float) + 1e-6)
            print(iou_map)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string(f'{scene_id[batch_idx]} 的平均IoU：{tmp_iou:.4f}')
            print('----------------------------')

            # 保存预测结果用于可视化
            filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for label in pred_label:
                    pl_save.write(f"{int(label)}\n")
            # 写入OBJ文件（带颜色）
            for i in range(whole_scene_label.shape[0]):
                color = g_label2color[pred_label[i]]       # 预测标签颜色
                color_gt = g_label2color[whole_scene_label[i]]  # 真实标签颜色
                if args.visual:
                    fout.write(f'v {whole_scene_data[i, 0]:f} {whole_scene_data[i, 1]:f} {whole_scene_data[i, 2]:f} {color[0]} {color[1]} {color[2]}\n')
                    fout_gt.write(f'v {whole_scene_data[i, 0]:f} {whole_scene_data[i, 1]:f} {whole_scene_data[i, 2]:f} {color_gt[0]} {color_gt[1]} {color_gt[2]}\n')
            if args.visual:
                fout.close()
                fout_gt.close()

        # 计算全局指标
        IoU = total_correct_class / (total_iou_deno_class.astype(np.float) + 1e-6)
        iou_per_class_str = '------- 类别IoU -------\n'
        for l in range(NUM_CLASSES):
            class_name = seg_label_to_cat[l]
            iou_per_class_str += f'类别 {class_name.ljust(14)} IoU：{total_correct_class[l] / total_iou_deno_class[l]:.3f}\n'
        log_string(iou_per_class_str)
        log_string(f'评估点云平均类别IoU：{np.mean(IoU):f}')
        log_string(f'评估整体场景点云平均类别准确率：{np.mean(total_correct_class / (np.array(total_seen_class, dtype=np.float) + 1e-6)):f}')
        log_string(f'评估整体场景点云准确率：{np.sum(total_correct_class) / (np.sum(total_seen_class) + 1e-6):f}')

        print("完成！")


if __name__ == '__main__':
    args = parse_args()
    main(args)