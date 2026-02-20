import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher_crowd

import numpy as np
import time
import math
# from .deform_conv_v2 import DeformConv2D
# from .block import VGGFeatureExtractor as get_Extractor


# copy from SRformer: https://github.com/HVision-NKU/SRFormer/blob/main/basicsr/archs/srformer_arch.py#L713
class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


# the network frmawork of the regression branch
class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        # self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act3 = nn.ReLU()

        # self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)

    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 2)


# the network frmawork of the classification branch
class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        # self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act3 = nn.ReLU()

        # self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, _ = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)

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


# class Restoration(nn.Module):
#     def __init__(self, input_nc, output_nc):
#         super(Restoration, self).__init__()
#
#         # batchx4x512x16x16 => batchx4x256x32x32
#         self.conv_1 = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(256),
#             nn.SiLU()
#         )
#
#         # batchx4x512x32x32 => batchx4x128x64x64
#         self.conv_2 = nn.Sequential(
#             nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(128),
#             nn.SiLU()
#         )
#
#         # batchx4x256x64x64 => batchx4x64x128x128
#         self.conv_3 = nn.Sequential(
#             nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(64),
#             nn.SiLU()
#         )
#
#         # batchx4x64x128x128 => batchx4x3x128x128
#         self.conv_4 = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(64, output_nc, 7),
#             nn.Tanh()
#         )
#
#     def forward(self, x, y, z):
#         """Standard forward"""
#
#         # batchx4x512x16x16 => batchx4x256x32x32
#         c1 = self.conv_1(x)
#
#         # batchx4x256x32x32 => batchx4x512x32x32
#         skip1_de = torch.cat((c1, y), 1)
#
#         # batchx4x512x32x32 => batchx4x128x64x64
#         c2 = self.conv_2(skip1_de)
#
#         # batchx4x128x64x64 => batchx4x256x64x64
#         skip2_de = torch.cat((c2, z), 1)
#
#         # batchx4x256x64x64 => batchx4x64x128x128
#         c3 = self.conv_3(skip2_de)
#
#         # batchx4x64x128x128=> batchx4x3x128x128
#         dehaze = self.conv_4(c3)
#
#         # norm to 0-1, add by wedream
#         # dehaze = dehaze/2+0.5
#
#         # print("dehaze(size):{}".format(dehaze.size()))
#         # end
#
#         return dehaze

class MultiScaleRestoration(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(MultiScaleRestoration, self).__init__()

        # batchx4x512x16x16 => batchx4x512x64x64
        # self.upscore4 = nn.UpsamplingBilinear2d(scale_factor=4)

        # batchx4x256x32x32 => batchx4x256x64x64
        self.upscore2 = nn.UpsamplingBilinear2d(scale_factor=2)

        # batchx4x(512+256+128)x64x64 => batchx4x64x128x128
        # self.conv_1 = nn.Sequential(
        #     nn.ConvTranspose2d(896, 64, 3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.SiLU()
        # )

        # batchx4x(256+128)x64x64 => batchx4x64x128x128
        self.conv_1 = nn.Sequential(
            nn.ConvTranspose2d(384, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )

        # batchx4x64x128x128 => batchx4x3x128x128
        self.conv_2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        )

    def forward(self, x, y, z):
        """Standard forward, x是最深最小的特征图"""
        # x => batchx4x512x16x16
        # y => batchx4x256x32x32
        # z => batchx4x128x64x64

        # batchx4x512x16x16 => batchx4x512x64x64
        # x4 = self.upscore4(x)
        # batchx4x256x32x32 => batchx4x256x64x64
        y2 = self.upscore2(y)

        # print("x4: {}".format(x4.size()))
        # print("y2: {}".format(y2.size()))
        # print("z: {}".format(z.size()))

        # fuze_xyz = torch.cat((x4, y2, z), 1)
        fuze_xyz = torch.cat((y2, z), 1)

        fuze_conv1  = self.conv_1(fuze_xyz)

        # batchx4x256x64x64 => batchx4x64x128x128
        dehaze = self.conv_2(fuze_conv1)

        # batchx4x64x128x128=> batchx4x3x128x128

        return dehaze



class Decoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(Decoder, self).__init__()

        # 将FPN中的卷积换成可变形卷积

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # self.P5_2 = DeformConv2D(feature_size, feature_size, kernel_size=3, stride=1, padding=1, modulation=True)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # self.P4_2 = DeformConv2D(feature_size, feature_size, kernel_size=3, stride=1, padding=1, modulation=True)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # self.P3_2 = DeformConv2D(feature_size, feature_size, kernel_size=3, stride=1, padding=1, modulation=True)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]


# the defenition of the P2PNet model
class P2PNet(nn.Module):
    def __init__(self, backbone, row=2, line=2, in_chans=3, upscale=1, embed_dim=96, num_feat=64):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2

        # add by wedream
        self.upscale = upscale
        num_in_ch = in_chans
        num_out_ch = in_chans
        # end

        # the number of all anchor points
        num_anchor_points = row * line

        self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        self.classification = ClassificationModel(num_features_in=256, \
                                                  num_classes=self.num_classes, \
                                                  num_anchor_points=num_anchor_points)

        self.anchor_points = AnchorPoints(pyramid_levels=[3, ], row=row, line=line)

        self.fpn = Decoder(256, 512, 512)

        # add by wedream
        # copy from TogetherNet: https://github.com/yz-wang/TogetherNet/blob/main/nets/yolo.py#L113
        self.restormer = MultiScaleRestoration(512, 3)
        # end

    # def forward(self, samples: NestedTensor, is_train=False):
    def forward(self, samples: NestedTensor, is_train=True):
        # get the backbone features
        # print("samples_size:{}".format(samples.size()))
        # samples_size=>torch.Size([4, 3, 128, 128])

        features = self.backbone(samples)  # list

        # print("features[0].size:{}".format(features[0].size()))=>torch.Size([4, 128, 64, 64])
        # print("features[1].size:{}".format(features[1].size()))=>torch.Size([4, 256, 32, 32])
        # print("features[2].size:{}".format(features[2].size()))=>torch.Size([4, 512, 16, 16])
        # print("features[3].size:{}".format(features[3].size()))=>torch.Size([4, 512, 8, 8])
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

        # train
        if is_train:
            # SR branch
            sr = self.restormer(features[2], features[1], features[0])
            # sr = samples
            # print("sr_size:{}".format(sr.size()))
            out = {'pred_logits': output_class, 'pred_points': output_coord, 'pred_sr': sr}
        # inference
        else:
            out = {'pred_logits': output_class, 'pred_points': output_coord}

        return out


class SetCriterion_Crowd(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, vgg_model=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            vgg_model: the vgg model to be calculated perception loss. add by wedream.
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
        # add by wedream
        self.vgg_model = vgg_model
        # end

    def loss_labels(self, outputs, targets, hr_imgs, indices, num_points):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_points(self, outputs, targets, hr_imgs, indices, num_points):

        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.mse_loss(src_points, target_points, reduction='none')

        losses = {}
        # losses['loss_point'] = loss_bbox.sum() / num_points
        # 将损失修改回来，梯度回传
        losses['loss_points'] = loss_bbox.sum() / num_points

        return losses

    def loss_sr(self, outputs, targets, hr_imgs, indices, num_points):

        assert 'pred_sr' in outputs
        # idx = self._get_src_permutation_idx(indices)
        # sr_imgs = outputs['pred_sr'][0] # 这里出错了
        sr_imgs = outputs['pred_sr']

        criterion = nn.MSELoss()
        # 切换成L1损失
        # criterion = nn.L1Loss()

        loss_sr = criterion(sr_imgs, hr_imgs)  # L1/L2 loss

        losses = {}
        # losses['loss_sr'] = loss_sr.sum()
        losses['loss_sr'] = loss_sr

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

    def get_loss(self, loss, outputs, targets, hr_imgs, indices, num_points, **kwargs):
        # loss_map = {
        #     'labels': self.loss_labels,
        #     'points': self.loss_points,
        # }
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
            'sr': self.loss_sr
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, hr_imgs, indices, num_points, **kwargs)
        # if loss == 'sr':
        #     return loss_map[loss](outputs, targets, hr_imgs, indices, num_points, self.vgg_model, **kwargs)
        # else:
        #     return loss_map[loss](outputs, targets, hr_imgs, indices, num_points, **kwargs)

    def forward(self, outputs, targets, hr_imgs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points'],
                   'pred_sr': outputs['pred_sr']}

        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, hr_imgs, indices1, num_boxes))

        return losses


# create the P2PNet model
# try add perception loss to loss_sr
def build25(args, training):
    print("build P2PNetV25!")
    print("build L2 Loss!")
    # treats persons as a single class
    num_classes = 1

    # add by wedream
    # device = torch.device('cpu')
    # device = torch.device('cuda')
    # vgg = get_Extractor().eval().to(device)  # 用于计算感知损失
    # print("load VGG successfully!")
    # end

    backbone = build_backbone(args)
    model = P2PNet(backbone, args.row, args.line)
    if not training:
        return model

    # weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef}
    # weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef, 'loss_sr': 0.01}
    # weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef, 'loss_sr': 0.1}
    weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef, 'loss_sr': args.sr_loss_coef}
    # losses = ['labels', 'points']
    losses = ['labels', 'points', 'sr']
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(num_classes, \
                                   matcher=matcher, weight_dict=weight_dict, \
                                   eos_coef=args.eos_coef, losses=losses)

    return model, criterion