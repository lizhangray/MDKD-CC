import argparse
import datetime
import random
import time
from pathlib import Path

"""
数据预处理，
有复原分支，
去雾数据集，
学习率统一为0.0001
我们通常使用run_test25.py进行测试
"""
import torch
from torch.utils.data import DataLoader, DistributedSampler

from crowd_datasets import build_dataset
# from engine4 import *
from engine6 import *
from models import build_model25 as build_model
import os
import warnings

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)
    # add by wedream
    parser.add_argument('--sr_loss_coef', default=10, type=float)
    # end
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    # dataset parameters
    parser.add_argument('--dataset_file', default='Hazy-ShanghaiTechRGBD-R')
    parser.add_argument('--data_root', default='.',
                        help='path where the dataset is')

    parser.add_argument('--output_dir', default='vis/231118-1-vis',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./ckpt',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./runs',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 尝试加载预训练权重，会不会好点
    parser.add_argument('--pretrain', default='weights/231118-1/best_mae.pth', help='load pretrained weight from checkpoint')
    # parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
    #                     help='start epoch')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser

# number of params: 21810131

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    # create the logging file

    # add by wedream
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # fix the seed for reproducibility
    # 固定随机种子
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # get the P2PNet model
    model, criterion = build_model(args, training=True)
    # move to GPU
    model.to(device)
    criterion.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # create the dataset
    loading_data = build_dataset(args=args)
    # create the training and valiation set
    train_set, val_set = loading_data(args.data_root)
    sampler_val = torch.utils.data.SequentialSampler(val_set)



    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)


    if args.pretrain != '':
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Load pretrained weight from {}! Initialize successfully!".format(args.pretrain))



    print("Start testing")
    start_time = time.time()
    # save the performance during the training
    mae = []
    mse = []
    # the logger writer
    # writer = SummaryWriter(args.tensorboard_dir)

    # step = 1
    # training starts here
    t1 = time.time()
    # 输出dehazed图和points
    result = test_crowd_no_overlap(model, data_loader_val, device, vis_dir=args.output_dir)
    # 可视化点坐标
    # result = evaluate_crowd_no_overlap(model, data_loader_val, device, vis_dir=args.output_dir)
    # 输出点坐标
    # result = points_crowd_no_overlap(model, data_loader_val, device, vis_dir=args.output_dir)
    t2 = time.time()

    mae.append(result[0])
    mse.append(result[1])

    # print the evaluation results
    print('=======================================test=======================================')
    print("mae:", result[0], "mse:", result[1], "time:", t2 - t1, "best mae:", np.min(mae), )
    print('=======================================test=======================================')

    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)