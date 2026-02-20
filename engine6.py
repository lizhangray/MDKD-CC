# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import math
import os
import sys
from typing import Iterable

import torch

# import util.misc as utils
import util.misc2 as utils
from util.misc import NestedTensor
import numpy as np
import time
import torchvision.transforms as standard_transforms
import cv2
# from my_utils.evaluation import eval_game, eval_relative

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def vis(samples, targets, pred, vis_dir, des=None):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''
    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        # DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 2
        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for p in pred[idx]:
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name = targets[idx]['image_id']
        # save the visualized images
        if des is not None:
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_gt)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_pred)
        else:
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_gt)
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_pred.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_pred)

def vis2(samples, targets, pred, vis_dir, des=None):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''
    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        # DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 2
        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for p in pred[idx]:
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name = targets[idx]['image_id']
        # save the visualized images
        if des is not None:
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name),
                                                des, len(gts[idx]), len(pred[idx]))), sample_gt)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name),
                                                des, len(gts[idx]), len(pred[idx]))), sample_pred)
        else:
            # cv2.imwrite(
            #     os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
            #     sample_gt)
            cv2.imwrite(
                os.path.join(vis_dir, 'IMG_{}_gt{}_c{}.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_pred)


def vis3(samples, targets, pred, vis_dir, des=None):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''
    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        # DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        # size = 2
        # size = 4
        # size = 4
        size = 6
        draw_color = (0, 0, 255)  # (0, 255, 0)-green, (0, 0, 255)-red

        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for p in pred[idx]:
            # sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, draw_color, -1)

        name = targets[idx]['image_id']
        # save the visualized images
        if des is not None:
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name),
                                                des, len(gts[idx]), len(pred[idx]))), sample_gt)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name),
                                                des, len(gts[idx]), len(pred[idx]))), sample_pred)
        else:
            # cv2.imwrite(
            #     os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
            #     sample_gt)
            cv2.imwrite(
                os.path.join(vis_dir, 'IMG_{}_gt{}_c{}.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_pred)


def save_points(samples, targets, pred, vis_dir, des=None):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''
    print("pred: {}".format(pred))

    # draw one by one
    for idx in range(samples.shape[0]):
        # sample = restore_transform(samples[idx])
        # sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
        name = targets[idx]['image_id']
        np.save(os.path.join(vis_dir, "{}".format(int(name))), pred)
        # print(np.load(os.path.join(vis_dir, "{}.npy".format(int(name)))))
        # exit(0)

        # save the points


# the training routine
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # iterate all training samples
    for samples, hr_samples, targets in data_loader:
        samples = samples.to(device)
        # print("samples:{}".format(samples.size()))
        # print("hr_samples:{}".format(hr_samples))
        hr_samples = hr_samples.to(device)
        # print("hr_samples in loop:{}".format(hr_samples.size()))
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward
        outputs = model(samples)
        # calc the losses
        loss_dict = criterion(outputs, targets, hr_samples)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce all losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # backward
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # update logger
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# the inference routine
@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, vis_dir=None):
    model.eval()
    # print("evaluate_crowd_no_overlap!")
    if vis_dir:
        save_dir1 = os.path.join(vis_dir,'restoration')
        save_dir2 = os.path.join(vis_dir, 'crowd')

    if vis_dir:
        if not os.path.exists(save_dir1):
            os.makedirs(save_dir1)
        if not os.path.exists(save_dir2):
            os.makedirs(save_dir2)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    show_cnt = 1
    for samples, hr_samples, targets in data_loader:
        # print("calculate image {}".format(show_cnt))
        samples = samples.to(device)

        # add by wedream
        hr_samples = hr_samples.to(device)
        # end

        # outputs = model(samples)
        if show_cnt<=3:
            outputs = model.forward(samples, is_train=False)
            # outputs = model.forward(samples, is_train=True)
            # # add by wedream
            # lr0 = samples[0]
            # hr0 = hr_samples[0]
            # sr0 = outputs['pred_sr'][0]
        else:
            outputs = model.forward(samples, is_train=False)


        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]



        # 展示3张训练过程超分图
        # if show_cnt <= 3:
        if show_cnt <= -1:
            from crowd_datasets.SHHAD.loading_data import DeNormalize,save_image_tensor
            # 反归一化，恢复成原图像
            # denorm = DeNormalize(mean=[0.485, 0.456, 0.406],
            #                         std=[0.229, 0.224, 0.225])
            # img_denorm = denorm(hr0)
            # hr_denorm = denorm(sr0)

            # save_img = torch.cat((img_denorm.unsqueeze(0), hr_denorm.unsqueeze(0)), 1)

            save_image_tensor(lr0.unsqueeze(0), 'vis/vis{}_lr.png'.format(show_cnt))
            save_image_tensor(hr0.unsqueeze(0), 'vis/vis{}_hr.png'.format(show_cnt))
            save_image_tensor(sr0.unsqueeze(0), 'vis/vis{}_sr.png'.format(show_cnt))
            # if show_cnt == 1 :
            #     print("hr0: {}".format(hr0))
            #     print("sr0: {}".format(sr0))

            show_cnt += 1

            # end

        gt_cnt = targets[0]['point'].shape[0]
        # 0.5 is used by default
        threshold = 0.5

        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())
        # if specified, save the visualized images
        if vis_dir is not None: 
            # vis(samples, targets, [points], vis_dir)
            vis3(samples, targets, [points], vis_dir=save_dir2)
        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))
    # calc MAE, RMSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    return mae, mse

# the inference routine
@torch.no_grad()
def infer_for_mae_rmse(model, data_loader, device, vis_dir=None):
    model.eval()
    # print("evaluate_crowd_no_overlap!")
    if vis_dir:
        save_dir1 = os.path.join(vis_dir,'restoration')
        save_dir2 = os.path.join(vis_dir, 'crowd')

    if vis_dir:
        if not os.path.exists(save_dir1):
            os.makedirs(save_dir1)
        if not os.path.exists(save_dir2):
            os.makedirs(save_dir2)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    show_cnt = 1
    for samples, hr_samples, targets in data_loader:
        print("calculate image {}".format(show_cnt))
        samples = samples.to(device)

        # add by wedream
        hr_samples = hr_samples.to(device)
        # end

        # outputs = model(samples)
        outputs = model.forward(samples, is_train=False)

        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        show_cnt+=1

        gt_cnt = targets[0]['point'].shape[0]
        # 0.5 is used by default
        threshold = 0.5

        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())

        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))
    # calc MAE, RMSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    return mae, mse

# the inference routine
@torch.no_grad()
def points_crowd_no_overlap(model, data_loader, device, vis_dir=None):
    model.eval()
    # print("evaluate_crowd_no_overlap!")
    save_dir1 = os.path.join(vis_dir,'restoration')
    save_dir2 = os.path.join(vis_dir, 'crowd')
    save_dir3 = os.path.join(vis_dir, 'points')

    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)
    if not os.path.exists(save_dir3):
        os.makedirs(save_dir3)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    show_cnt = 1
    for samples, hr_samples, targets in data_loader:
        samples = samples.to(device)

        # add by wedream
        hr_samples = hr_samples.to(device)
        # end

        # outputs = model(samples)
        # if show_cnt<=3:
        if show_cnt<=-1:
            outputs = model.forward(samples, is_train=True)
            # add by wedream
            lr0 = samples[0]
            hr0 = hr_samples[0]
            sr0 = outputs['pred_sr'][0]
        else:
            outputs = model.forward(samples, is_train=False)


        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]



        # 展示3张训练过程超分图
        # if show_cnt <= 3:
        if show_cnt <= -1:
            from crowd_datasets.SHHAD.loading_data import DeNormalize,save_image_tensor
            # 反归一化，恢复成原图像
            # denorm = DeNormalize(mean=[0.485, 0.456, 0.406],
            #                         std=[0.229, 0.224, 0.225])
            # img_denorm = denorm(hr0)
            # hr_denorm = denorm(sr0)

            # save_img = torch.cat((img_denorm.unsqueeze(0), hr_denorm.unsqueeze(0)), 1)

            save_image_tensor(lr0.unsqueeze(0), 'vis/vis{}_lr.png'.format(show_cnt))
            save_image_tensor(hr0.unsqueeze(0), 'vis/vis{}_hr.png'.format(show_cnt))
            save_image_tensor(sr0.unsqueeze(0), 'vis/vis{}_sr.png'.format(show_cnt))
            # if show_cnt == 1 :
            #     print("hr0: {}".format(hr0))
            #     print("sr0: {}".format(sr0))

            show_cnt += 1

            # end

        gt_cnt = targets[0]['point'].shape[0]
        # 0.5 is used by default
        threshold = 0.5

        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())
        # if specified, save the visualized images
        if vis_dir is not None:
            # vis(samples, targets, [points], vis_dir)
            # vis3(samples, targets, [points], vis_dir=save_dir2)
            save_points(samples, targets, points, vis_dir=save_dir3)
        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))
    # calc MAE, RMSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    return mae, mse


# the inference routine
@torch.no_grad()
def evaluate_crowd_no_overlap2(model, data_loader, device, vis_dir=None):
    model.eval()

    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []

    # Iterate over data.
    game_list = [0, 0, 0, 0]
    mse_list = [0, 0, 0, 0]
    total_relative_error = 0

    show_cnt = 1
    for samples, hr_samples, targets in data_loader:
        samples = samples.to(device)

        # add by wedream
        hr_samples = hr_samples.to(device)
        # end

        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        # add by wedream
        hr0 = hr_samples[0]
        sr0 = outputs['pred_sr'][0]

        #
        print("show_cnt: {}".format(show_cnt))
        show_cnt += 1


            # end

        gt_cnt = targets[0]['point'].shape[0]
        # 0.5 is used by default
        threshold = 0.5

        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())

        # add by wedream
        # for L in range(4):
        #     abs_error, square_error = eval_game(outputs_points[outputs_scores > threshold], targets[0]['point'], L)
        #     game_list[L] += abs_error
        #     mse_list[L] += square_error
        # relative_error = eval_relative(outputs_points[outputs_scores > threshold], targets[0]['point'])
        # total_relative_error += relative_error
        # end


        # if specified, save the visualized images
        if vis_dir is not None:
            vis(samples, targets, [points], vis_dir)
        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))
    # calc MAE, RMSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    # add by wedream
    # N = len(data_loader)
    # game_l = [m / N for m in game_list]
    # mse_l = [torch.sqrt(m / N) for m in mse_list]
    # total_relative_error = total_relative_error / N
    # # end
    # log_str = 'Test{}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
    #           'MSE {mse:.2f} Re {relative:.4f}, '. \
    #     format(N, game0=game_l[0], game1=game_l[1], game2=game_l[2], game3=game_l[3], mse=mse_l[0], relative=total_relative_error)
    #
    # print(log_str)

    return mae, mse

# the inference routine
@torch.no_grad()
def test_crowd_no_overlap(model, data_loader, device, vis_dir=None):
    #from crowd_datasets.SHHAD.loading_data import save_image_tensor
    from crowd_datasets.Rainy_JHU.loading_data2 import save_image_tensor
    model.eval()

    save_dir1 = os.path.join(vis_dir,'restoration')
    save_dir2 = os.path.join(vis_dir, 'crowd')
    save_dir3 = os.path.join(vis_dir, 'points')

    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)
    if not os.path.exists(save_dir3):
        os.makedirs(save_dir3)
    #addbyYAO
    image_metrics = []
    # run inference on all images to calc MAE
    maes = []
    mses = []
    show_cnt = 1
    for samples, hr_samples, targets in data_loader:
        print("calculate image {}".format(show_cnt))
        samples = samples.to(device)

        # add by wedream
        hr_samples = hr_samples.to(device)
        # end

        # outputs = model(samples)
        outputs = model.forward(samples, is_train=True)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        # add by wedream
        lr0 = samples[0]
        hr0 = hr_samples[0]
        sr0 = outputs['pred_sr'][0]

        # save_image_tensor(lr0.unsqueeze(0), 'vis/vis{}_lr.png'.format(show_cnt))
        # save_image_tensor(hr0.unsqueeze(0), 'vis/vis{}_hr.png'.format(show_cnt))
        save_image_tensor(sr0.unsqueeze(0), os.path.join(save_dir1,'IMG_{}_sr.png'.format(int(targets[0]['image_id']))))

        gt_cnt = targets[0]['point'].shape[0]
        # 0.5 is used by default
        threshold = 0.5

        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())
        # if specified, save the visualized images
        if vis_dir is not None:
            vis2(samples, targets, [points], vis_dir=save_dir2)
            save_points(samples, targets, points, vis_dir=save_dir3)
        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)

        #addbyYAO 保存每张图片的指标
        image_id = int(targets[0]['image_id'])
        image_metrics.append({
            'image_id': image_id,
            'predicted_count': predict_cnt,
            'ground_truth_count': gt_cnt,
            'mae': float(mae),
            'mse': float(mse)
        })

        maes.append(float(mae))
        mses.append(float(mse))
        show_cnt += 1
    #addbyYAO 保存每张图片的指标到文件
    if vis_dir is not None:
        import json
        metrics_file = os.path.join(vis_dir, 'image_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(image_metrics, f, indent=2)
        
    # 同时保存为CSV格式
    import csv
    csv_file = os.path.join(vis_dir, 'image_metrics.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'predicted_count', 'ground_truth_count', 'mae', 'mse'])
        for metric in image_metrics:
            writer.writerow([
                metric['image_id'],
                metric['predicted_count'],
                metric['ground_truth_count'],
                metric['mae'],
                metric['mse']
            ])

    # calc MAE, RMSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    return mae, mse



# 新的指标！！！
@torch.no_grad()
def evaluate_crowd_counting_and_loc(model, data_loader, device, threshold=0.5, vis_dir=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    nMAE = 0
    intervals = {}
    tp_sum_4 = 0
    gt_sum = 0
    et_sum = 0
    tp_sum_8 = 0
    # for ct, (samples, targets) in enumerate(data_loader):
    for ct, (samples, hr_samples, targets) in enumerate(data_loader):
        samples = samples.to(device)

        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]

        gt_cnt = targets[0]['point'].shape[0]
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())

        # if specified, save the visualized images
        if vis_dir is not None:
            vis(samples, targets, [points], vis_dir)

        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        print("mae:{}, mse:{}".format(mae, mse))
        maes.append(float(mae))
        mses.append(float(mse))

        # nMAE += mae/gt_cnt
        interval = int(gt_cnt / 250)
        if interval not in intervals:
            intervals[interval] = [mae / gt_cnt]
        else:
            intervals[interval].append(mae / gt_cnt)

        tp_4 = utils.compute_tp(points, targets[0]['point'], 4)
        tp_8 = utils.compute_tp(points, targets[0]['point'], 8)
        tp_sum_4 += tp_4
        gt_sum += gt_cnt
        et_sum += predict_cnt
        tp_sum_8 += tp_8

    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    # nMAE /= len(data_loader)
    ap_4 = tp_sum_4 / float(et_sum + 1e-10)
    ar_4 = tp_sum_4 / float(gt_sum + 1e-10)
    f1_4 = 2 * ap_4 * ar_4 / (ap_4 + ar_4 + 1e-10)
    ap_8 = tp_sum_8 / float(et_sum + 1e-10)
    ar_8 = tp_sum_8 / float(gt_sum + 1e-10)
    f1_8 = 2 * ap_8 * ar_8 / (ap_8 + ar_8 + 1e-10)
    local_result = {'ap_4': ap_4, 'ar_4': ar_4, 'f1_4': f1_4, 'ap_8': ap_8, 'ar_8': ar_8, 'f1_8': f1_8}
    return mae, mse, local_result