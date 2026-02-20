import copy
import os
import random

import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from models.backbone_rep import build_backbone, Backbone_VGG
from models.vgg_rep import RepBlock
from models.matcher import build_matcher_crowd


# number of params: 767648
import numpy as np
import time
import math
# from .deform_conv_v2 import DeformConv2D
# from .block import VGGFeatureExtractor as get_Extractor

"""
Similarity-Preserving Knowledge Distillation
"""


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def relation_similarity(fm_s, fm_t):
    fm_s = fm_s.view(fm_s.size(0), -1)
    G_s = torch.mm(fm_s, fm_s.t())
    norm_G_s = F.normalize(G_s, p=2, dim=1)

    fm_t = fm_t.view(fm_t.size(0), -1)
    G_t = torch.mm(fm_t, fm_t.t())
    norm_G_t = F.normalize(G_t, p=2, dim=1)

    loss = F.mse_loss(norm_G_s, norm_G_t)

    return loss

def cosine_similarity(stu_map, tea_map):
    # reduction = 'mean'
    # similiar = 1-F.cosine_similarity(stu_map, tea_map, dim=1)
    similiar = 1 - F.cosine_similarity(stu_map, tea_map, dim=1)
    # loss = similiar.sum()
    loss = torch.mean(similiar)
    return loss

# the network frmawork of the regression branch
# the network frmawork of the regression branch
class RegressionModel(nn.Module):
    # def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
    # def __init__(self, num_features_in, num_anchor_points=4, feature_size=64, transform=False):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=128, transform=False, deploy=False):
        super(RegressionModel, self).__init__()

        self.transform=transform
        self.deploy = deploy

        # self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv1 = RepBlock(num_features_in, feature_size, deploy=deploy)
        self.act1 = nn.ReLU()

        if self.transform:
            # self.reg_transform1_0 = feature_transform(feature_size, feature_size*4)
            self.reg_transform1_0 = feature_transform(feature_size, feature_size * 2)

        # self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv2 = RepBlock(feature_size, feature_size, deploy=deploy)
        self.act2 = nn.ReLU()

        if self.transform:
            # self.reg_transform2_0 = feature_transform(feature_size, feature_size*4)
            self.reg_transform2_0 = feature_transform(feature_size, feature_size * 2)

        # self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)
        self.output = RepBlock(feature_size, num_anchor_points * 2, deploy=deploy)


    # sub-branch forward
    def forward(self, x):
        features_list = []

        out = self.conv1(x)
        out = self.act1(out)

        if self.transform:
            features_list.append(self.reg_transform1_0(out))

        out = self.conv2(out)
        out = self.act2(out)

        if self.transform:
            features_list.append(self.reg_transform2_0(out))

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        if self.transform:
            return features_list, out.contiguous().view(out.shape[0], -1, 2)

        return out.contiguous().view(out.shape[0], -1, 2)


# the network frmawork of the classification branch
class ClassificationModel(nn.Module):
    # def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=256):
    # def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=64, transform=False):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=128, transform=False, deploy=False):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points
        self.transform = transform
        self.deploy = deploy

        # self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv1 = RepBlock(num_features_in, feature_size, deploy=deploy)
        self.act1 = nn.ReLU()

        if self.transform:
            # self.cls_transform1_0 = feature_transform(feature_size, feature_size*4)
            self.cls_transform1_0 = feature_transform(feature_size, feature_size*2)

        # self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv2 = RepBlock(feature_size, feature_size, deploy=deploy)
        self.act2 = nn.ReLU()

        if self.transform:
            # self.cls_transform2_0 = feature_transform(feature_size, feature_size*4)
            self.cls_transform2_0 = feature_transform(feature_size, feature_size*2)

        # self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output = RepBlock(feature_size, num_anchor_points * num_classes, deploy=deploy)
        self.output_act = nn.Sigmoid()

    # sub-branch forward
    def forward(self, x):
        features_list = []

        out = self.conv1(x)
        out = self.act1(out)

        if self.transform:
            features_list.append(self.cls_transform1_0(out))

        out = self.conv2(out)
        out = self.act2(out)

        if self.transform:
            features_list.append(self.cls_transform2_0(out))

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, _ = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)

        if self.transform:
            return features_list, out2.contiguous().view(x.shape[0], -1, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


# generate the reference points in grid layout
def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points


# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points


# this class generate all reference points on all pyramid levels
class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        # get reference points for each level
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2 ** p, row=self.row, line=self.line)
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx], anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)

        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        # send reference points to device
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))


class Decoder(nn.Module):
    # def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
    # def __init__(self, C3_size, C4_size, C5_size, feature_size=64):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=128):
        super(Decoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        # print("C3.size: {}".format(C3.size()))
        # print("C4.size: {}".format(C4.size()))
        # print("C5.size: {}".format(C5.size()))
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]



def feature_transform(inp, oup):
    conv2d = nn.Conv2d(inp, oup, kernel_size=1)  # no padding
    relu = nn.ReLU(inplace=True)
    layers = []
    layers += [conv2d, relu]
    return nn.Sequential(*layers)


# the defenition of the P2PNet model
class P2PNet(nn.Module):
    def __init__(self, backbone, row=2, line=2, transform=False, deploy=False):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        self.transform = transform
        self.deploy = deploy
        # the number of all anchor points
        num_anchor_points = row * line

        if self.transform:
            self.backnone_transform0_0 = feature_transform(64, 128)
            self.backnone_transform1_0 = feature_transform(128, 256)
            self.backnone_transform2_0 = feature_transform(256, 512)
            self.backnone_transform3_0 = feature_transform(256, 512)

        # self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        # self.regression = RegressionModel(num_features_in=64, num_anchor_points=num_anchor_points, transform=self.transform)
        # self.regression = RegressionModel(num_features_in=128, num_anchor_points=num_anchor_points, transform=self.transform)
        self.regression = RegressionModel(num_features_in=128, num_anchor_points=num_anchor_points, transform=self.transform, deploy=deploy)
        # self.classification = ClassificationModel(num_features_in=256, \
        # self.classification = ClassificationModel(num_features_in=64, \
        self.classification = ClassificationModel(num_features_in=128, \
                                                  num_classes=self.num_classes, \
                                                  num_anchor_points=num_anchor_points, \
                                                  transform=self.transform, \
                                                  deploy=deploy)

        self.anchor_points = AnchorPoints(pyramid_levels=[3, ], row=row, line=line)

        # self.fpn = Decoder(256, 512, 512)
        # self.fpn = Decoder(64, 128, 128)
        self.fpn = Decoder(128, 256, 256)
        if self.transform:
            # self.fpn_transform1_0 = feature_transform(128, 128*4)
            self.fpn_transform1_0 = feature_transform(128, 128*2)


    def forward(self, samples: NestedTensor):
        # get the backbone features
        # print("samples_size:{}".format(samples.size()))
        # samples_size=>torch.Size([4, 3, 128, 128])

        if self.transform:
            features = self.backbone(samples)  # list

            # print("features[0].size:{}".format(features[0].size()))=>torch.Size([4, 128, 64, 64])
            # print("features[1].size:{}".format(features[1].size()))=>torch.Size([4, 256, 32, 32])
            # print("features[2].size:{}".format(features[2].size()))=>torch.Size([4, 512, 16, 16])
            # print("features[3].size:{}".format(features[3].size()))=>torch.Size([4, 512, 8, 8])
            # 以上四个分别对应VGG四个特征图
            transform_features0 = self.backnone_transform0_0(features[0])
            transform_features1 = self.backnone_transform1_0(features[1])
            transform_features2 = self.backnone_transform2_0(features[2])
            transform_features3 = self.backnone_transform3_0(features[3])

            # forward the feature pyramid
            features_fpn = self.fpn([features[1], features[2], features[3]])
            transform_fpn1 = self.fpn_transform1_0(features_fpn[1])

            batch_size = features[0].shape[0]
            # run the regression and classification branch
            regression_output = self.regression(features_fpn[1])  # 8x
            reg_features, regression = regression_output[0], regression_output[1] * 100
            cls_features, classification = self.classification(features_fpn[1])
            anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
            # decode the points as prediction
            output_coord = regression + anchor_points
            output_class = classification
            out = {'pred_logits': output_class, 'pred_points': output_coord, \
                   'pred_backbones': [transform_features0, transform_features1, transform_features2, transform_features3], \
                   'pred_fpn': transform_fpn1, \
                   'reg_features': reg_features, 'cls_features': cls_features}

            return out
        else:

            features = self.backbone(samples)  # list

            # print("features[0].size:{}".format(features[0].size()))=>torch.Size([4, 128, 64, 64])
            # print("features[1].size:{}".format(features[1].size()))=>torch.Size([4, 256, 32, 32])
            # print("features[2].size:{}".format(features[2].size()))=>torch.Size([4, 512, 16, 16])
            # print("features[3].size:{}".format(features[3].size()))=>torch.Size([4, 512, 8, 8])
            # print("features[0].size:{}".format(features[0].size())) # =>torch.Size([4, 128, 64, 64])
            # print("features[1].size:{}".format(features[1].size())) # =>torch.Size([4, 256, 32, 32])
            # print("features[2].size:{}".format(features[2].size())) # =>torch.Size([4, 512, 16, 16])
            # print("features[3].size:{}".format(features[3].size())) # =>torch.Size([4, 512, 8, 8])
            # 以上四个分别对应VGG四个特征图

            # forward the feature pyramid
            features_fpn = self.fpn([features[1], features[2], features[3]])

            batch_size = features[0].shape[0]
            # run the regression and classification branch
            regression = self.regression(features_fpn[1]) * 100  # 8x
            classification = self.classification(features_fpn[1])
            anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
            # decode the points as prediction
            output_coord = regression + anchor_points
            output_class = classification
            out = {'pred_logits': output_class, 'pred_points': output_coord}

            return out


class SetCriterion_Crowd(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points, teachers, indices2, num_points2, teachers2targets):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        # for t, (_, J) in zip(targets, indices):
        #     print("J1: {}".format(J))
        #     print("[t[labels][J]1: {}".format(t["labels"][J]))
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        # print("target_classes_o: {}".format(target_classes_o))
        # print("target_classes: {}".format(target_classes))
        target_classes[idx] = target_classes_o

        loss_ce_hard = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        # add by wedream
        idx2 = self._get_src_permutation_idx(indices2)
        # for t, (_, J) in zip(teachers2targets, indices2):
        #     print("J2: {}".format(J))
        #     print("[t[labels][J]2: {}".format(t["labels"][J]))
        target_classes_o2 = torch.cat([t["labels"][J] for t, (_, J) in zip(teachers2targets, indices2)])
        target_classes2 = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        # target_classes2 = torch.full(src_logits.shape[:2], 0,
        #                              dtype=torch.float32, device=src_logits.device)
        # print("target_classes_o2: {}".format(target_classes_o2))
        # print("target_classes2: {}".format(target_classes2))
        target_classes2[idx2] = target_classes_o2

        loss_ce_soft = F.cross_entropy(src_logits.transpose(1, 2), target_classes2, self.empty_weight)

        # losses = {'loss_ce': loss_ce}
        alpha = 0.9
        losses = {'loss_ce': alpha * loss_ce_soft + (1 - alpha) * loss_ce_hard}

        return losses

    def loss_points(self, outputs, targets, indices, num_points, teachers, indices2, num_points2, teachers2targets):

        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.mse_loss(src_points, target_points, reduction='none')

        # add by wedream
        idx2 = self._get_src_permutation_idx(indices2)
        src_points2 = outputs['pred_points'][idx2]
        target_points2 = torch.cat([t['point'][i] for t, (_, i) in zip(teachers2targets, indices2)], dim=0)

        loss_bbox2 = F.mse_loss(src_points2, target_points2, reduction='none')

        losses = {}
        # losses['loss_point'] = loss_bbox.sum() / num_points
        # 将损失修改回来，梯度回传
        loss_hard = loss_bbox.sum()/num_points
        loss_soft = loss_bbox2.sum()/num_points2
        alpha = 0.9
        losses['loss_points'] = alpha*loss_soft + (1-alpha)*loss_hard

        # losses['loss_points'] = (loss_bbox.sum()+loss_bbox2.sum()) / (num_points+num_points2)

        return losses

    def loss_features(self, outputs, targets, indices, num_points, teachers, indices2, num_points2, teachers2targets):

        # print("outputs:{}".format(outputs))
        assert 'pred_backbones' in outputs
        # idx = self._get_src_permutation_idx(indices)
        # sr_imgs = outputs['pred_sr'][0] # 这里出错了

        pred_backbones0 = teachers['pred_backbones']
        pred_fpn0 = teachers['pred_fpn']
        reg_features0 = teachers['reg_features']
        cls_features0 = teachers['cls_features']

        pred_backbones1 = outputs['pred_backbones']
        pred_fpn1 = outputs['pred_fpn']
        reg_features1 = outputs['reg_features']
        cls_features1 = outputs['cls_features']
        # criterion = nn.MSELoss()
        # 切换成L1损失
        # criterion = nn.L1Loss()

        # print("f1.size, f2.size: {}, {}".format(reg_features0[0].shape, reg_features1[0].shape))

        #############################
        #  feature-based knowledge  #
        #############################
        loss_b0 = cosine_similarity(pred_backbones0[0], pred_backbones1[0])
        loss_b1 = cosine_similarity(pred_backbones0[1], pred_backbones1[1])
        loss_b2 = cosine_similarity(pred_backbones0[2], pred_backbones1[2])
        loss_b3 = cosine_similarity(pred_backbones0[3], pred_backbones1[3])

        loss_fpn = cosine_similarity(pred_fpn0, pred_fpn1)
        loss_reg0 = cosine_similarity(reg_features0[0], reg_features1[0])
        loss_reg1 = cosine_similarity(reg_features0[1], reg_features1[1])
        loss_cls0 = cosine_similarity(cls_features0[0], cls_features1[0])
        loss_cls1 = cosine_similarity(cls_features0[1], cls_features1[1])

        loss_f = loss_b0+loss_b1+loss_b2+loss_b3+loss_fpn+loss_reg0+loss_reg1+loss_cls0+loss_cls1
        loss_f = loss_f/9.

        #############################
        #  relation-based knowledge #
        #############################
        loss_b0_r = relation_similarity(pred_backbones0[0], pred_backbones1[0])
        loss_b1_r = relation_similarity(pred_backbones0[1], pred_backbones1[1])
        loss_b2_r = relation_similarity(pred_backbones0[2], pred_backbones1[2])
        loss_b3_r = relation_similarity(pred_backbones0[3], pred_backbones1[3])

        loss_fpn_r = relation_similarity(pred_fpn0, pred_fpn1)
        loss_reg0_r = relation_similarity(reg_features0[0], reg_features1[0])
        loss_reg1_r = relation_similarity(reg_features0[1], reg_features1[1])
        loss_cls0_r = relation_similarity(cls_features0[0], cls_features1[0])
        loss_cls1_r = relation_similarity(cls_features0[1], cls_features1[1])

        loss_r = loss_b0_r+loss_b1_r+loss_b2_r+loss_b3_r+loss_fpn_r+loss_reg0_r+loss_reg1_r+loss_cls0_r+loss_cls1_r
        loss_r = loss_r / 9.

        # loss_sr = criterion(sr_imgs, hr_imgs)  # L1/L2 loss

        beta = 0.5
        losses = {}
        # losses['loss_sr'] = loss_sr.sum()
        # losses['loss_features'] = loss_f/9

        losses['loss_features'] = (1.-beta) * loss_f + beta*loss_r

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, teachers, indices2, num_boxes2, teachers2targets, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
            'features': self.loss_features,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, teachers, indices2, num_boxes2, teachers2targets,  **kwargs)

    def forward(self, outputs, targets, teachers):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points'], 'pred_backbones': outputs['pred_backbones'], \
                   'pred_fpn': outputs['pred_fpn'], 'reg_features': outputs['reg_features'], 'cls_features': outputs['cls_features']}

        indices1 = self.matcher(output1, targets)
        # print("targets: {}".format(targets))
        # print("targets['labels']: {}".format(targets['labels']))
        # print("targets['point']: {}".format(targets['point']))
        # add by wedream
        # teachers['labels'] = teachers['pred_logits'] # 未经过Softmax
        teachers['labels'] = torch.nn.functional.softmax(teachers['pred_logits'], -1)[:, :, 1] # 取置信度高于0.5的作Soft Target
        # print("teachers['labels']: {}".format(teachers['labels']))
        # idx = torch.nonzero(teachers['labels'] > 0.5).reshape(-1)  # idx是列表
        threshold = 0.5
        idxs = [torch.nonzero(t_label > threshold).reshape(-1) for t_label in teachers['labels']]  # idx是列表
        teachers['labels'] = [teachers['labels'][_i][idxs[_i]] for _i in range(len(idxs))]
        # print("teachers['labels']: {}".format(teachers['labels']))
        teachers['point'] = teachers['pred_points']
        # print("teachers['point']: {}".format(teachers['point']))
        teachers['point'] = [teachers['point'][_i][idxs[_i]] for _i in range(len(idxs))]
        # print("teachers['point']: {}".format(teachers['point']))

        teachers2targets = [{'point':teachers['point'][_i], 'image_id':targets[_i]['image_id'], 'labels':[1]*len(teachers['labels'][_i])} for _i in range(len(idxs))]
        # teachers = [{'point':teachers['point'][_i], 'image_id':targets[_i]['image_id'], 'labels':torch.from_numpy(np.array([1 for _ in range(len(teachers['labels'][_i]))]))} for _i in range(len(idxs))]
        # teachers = [{'point':teachers['point'][_i], 'image_id':targets[_i]['image_id'], 'labels':torch.as_tensor([1 for _ in range(len(teachers['labels'][_i]))], dtype=torch.long, device=next(iter(output1.values())).device)} for _i in range(len(idxs))]
        # torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)

        # print("teachers2targets: {}".format(teachers2targets))
        # teachers = [{'point':teachers['point'][_i], 'image_id':targets[_i]['image_id'], 'labels':teachers['labels'][_i]} for _i in range(len(idxs))]
        # teachers = teachers2targets
        # print("teachers: {}".format(teachers))
        # print("teachers_labels: {}".format(teachers[0]['labels']))
        # print("targets_labels: {}".format(targets[0]['labels']))



        # indices2 = self.matcher(output1, teachers)
        indices2 = self.matcher(output1, teachers2targets)
        # end
        teachers2targets = [{'point':teachers['point'][_i], 'image_id':targets[_i]['image_id'], 'labels':torch.as_tensor([1 for _ in range(len(teachers['labels'][_i]))], dtype=torch.long, device=next(iter(output1.values())).device)} for _i in range(len(idxs))]

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        # num_points2 = sum(len(t["labels"]) for t in teachers)
        num_points2 = sum(len(t["labels"]) for t in teachers2targets)
        num_points2 = torch.as_tensor([num_points2], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points2)
        num_boxes2 = torch.clamp(num_points2 / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            # losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes, teachers))
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes, teachers, indices2, num_boxes2, teachers2targets))

        return losses

# create the P2PNet model
# try add perception loss to loss_sr
def build17_KD2_repvgg2_hinton_SP(args, training, deploy=False):
    print("build P2PNetV17_KD2_repvgg2_hinton_SP!")
    # print("build L1 Loss!")
    # treats persons as a single class
    num_classes = 1

    # add by wedream
    # device = torch.device('cpu')
    # device = torch.device('cuda')
    # vgg = get_Extractor().eval().to(device)  # 用于计算感知损失
    # print("load VGG successfully!")
    # end

    # add by wedream
    args.deploy = deploy
    # end

    backbone = build_backbone(args)
    model = P2PNet(backbone, args.row, args.line, transform=args.kd, deploy=deploy)
    if not training:
        return model

    # weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef, 'loss_features': 1}
    # weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef, 'loss_features': 0.1}
    weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef, 'loss_features': args.feat_loss_coef}
    losses = ['labels', 'points', 'features']
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(num_classes, \
                                matcher=matcher, weight_dict=weight_dict, \
                                eos_coef=args.eos_coef, losses=losses)

    return model, criterion


def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


if __name__ == '__main__':
    # backbone = Backbone_VGG('vgg16_bn_lite2', True)
    # model = P2PNet(backbone, 2, 2, transform=True)
    # print(model)

    seed_everything(2024)

    img = torch.ones((8, 3, 128, 128))
    backbone = Backbone_VGG('vgg16_bn_lite2', True, deploy=False)
    model = P2PNet(backbone, 2, 2, transform=True, deploy=False)
    # print(model)
    # exit(0)
    out = model(img)
    state_dict = model.state_dict()
    torch.save(state_dict, 'p2pnet.pth')
    print("pred_points: {}".format(out['pred_points'][0][0]))  # [19.4498, 33.2658]
    # # 模型转为部署模式保存
    # deploy_model = repvgg_model_convert(model, save_path='p2pnet_v17_kd2_deploy.pth', do_copy=False)
    deploy_model = repvgg_model_convert(model, save_path='p2pnet_deploy.pth')


    img = torch.ones((8, 3, 128, 128))
    backbone = Backbone_VGG('vgg16_bn_lite2', True, deploy=False)
    model = P2PNet(backbone, 2, 2, transform=True, deploy=False)
    model.load_state_dict(torch.load('p2pnet.pth', map_location='cpu'), strict=True)
    out = model(img)
    state_dict = model.state_dict()
    # torch.save(state_dict, 'p2pnet_v17_kd2.pth')
    print("pred_points: {}".format(out['pred_points'][0][0]))  # [19.4498, 33.2658]


    img = torch.ones((8, 3, 128, 128))
    backbone = Backbone_VGG('vgg16_bn_lite2', True, deploy=True)
    model = P2PNet(backbone, 2, 2, transform=True, deploy=True)
    # model = P2PNet(backbone, 2, 2, transform=False, deploy=True)
    model.load_state_dict(torch.load('p2pnet_deploy.pth', map_location='cpu'), strict=True)
    out = model(img)
    print("pred_points: {}".format(out['pred_points'][0][0])) # [19.4498, 33.2658]


    # pred_points: tensor([19.4498, 33.2658], grad_fn=<SelectBackward>)
