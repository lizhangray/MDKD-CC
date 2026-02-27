import argparse
import datetime
import random
import time
from pathlib import Path

"""
数据预处理，
无复原分支，
去雾数据集，
学习率统一为0.0001
"""
import torch
from torch.utils.data import DataLoader, DistributedSampler

from crowd_datasets import build_dataset
# from engine import *
from engine import *
from models import build_model17_lite2_repvgg2 as build_model
# from models.p2pnet_v17_lite2_repvgg2 import build17_lite2_repvgg2 as build_model
import os
import warnings

warnings.filterwarnings('ignore')

# part A
"""
=======================================test=======================================
mae: 65.97802197802197 mse: 111.7383619821838 time: 475.50138092041016 best mae: 65.97802197802197
=======================================test=======================================

=======================================Rep=======================================
mae: 65.97802197802197 mse: 111.7383619821838 time: 213.13700032234192 best mae: 65.97802197802197
=======================================test=======================================
"""
# part B
"""
=======================================test=======================================
mae: 7.984177215189874 mse: 13.523888849744854 time: 969.0052213668823 best mae: 7.984177215189874
=======================================test=======================================
"""


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn_lite', type=str,
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
    parser.add_argument('--dataset_file', default='SHHAFv2')
    parser.add_argument('--data_root', default='.',
                        help='path where the dataset is')

    parser.add_argument('--output_dir', default='vis/output_240403-lite2-kd2-repvgg2-02',
                        help='path where to save, empty for no saving')


    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 尝试加载预训练权重，会不会好点
    parser.add_argument('--pretrain', default='weights/240403-lite2-kd2-repvgg2-02/best_mae.pth', help='load pretrained weight from checkpoint')
    # parser.add_argument('--pretrain', default='weights/240402-lite2-repvgg2-02/best_mae_deploy.pth', help='load pretrained weight from checkpoint')


    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser


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
    # model, criterion = build_model(args, training=True)
    # model, criterion = build_model(args, training=True, deploy=True)
    from models.p2pnet_v17_KD4_repvgg2_hinton_SP import P2PNet
    from models.backbone_rep import Backbone_VGG

    # backbone = Backbone_VGG('vgg16_bn_lite2', True, deploy=True)
    # model = P2PNet(backbone, 2, 2, transform=False, deploy=True)

    # backbone = Backbone_VGG('vgg16_bn_lite2', True, deploy=False)
    backbone = Backbone_VGG('vgg16_bn_lite', True, deploy=False)
    model = P2PNet(backbone, 2, 2, transform=True, deploy=False)

    # model.transform=False
    # move to GPU
    model.to(device)
    # criterion.to(device)

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
        # model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        # model_without_ddp.load_state_dict(checkpoint, strict=True)
        print("Load pretrained weight from {}! Initialize successfully!".format(args.pretrain))

    # Re-parameterization
    from models.p2pnet_v17_KD2_repvgg2 import repvgg_model_convert
    repvgg_model_convert(model_without_ddp, save_path=args.pretrain.replace('.pth','_deploy.pth'))
    # print(args.pretrain.replace('.pth','_deploy.pth'))
    print("Re-parameterize weight from {} successfully!".format(args.pretrain))

    # backbone = Backbone_VGG('vgg16_bn_lite2', True, deploy=True)
    backbone = Backbone_VGG('vgg16_bn_lite', True, deploy=True)
    model = P2PNet(backbone, 2, 2, transform=True, deploy=True)
    model_without_ddp = model
    checkpoint_rep = torch.load(args.pretrain.replace('.pth','_deploy.pth'), map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint_rep, strict=True)
    model = model_without_ddp.to(device)



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
    result = test_crowd_no_overlap(model, data_loader_val, device, vis_dir=args.output_dir)
    # 可视化
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