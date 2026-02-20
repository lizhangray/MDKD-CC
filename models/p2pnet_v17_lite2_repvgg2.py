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
在骨干网络使用了重参数化技巧
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
            self.reg_transform1_0 = feature_transform(feature_size, feature_size*4)

        # self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv2 = RepBlock(feature_size, feature_size, deploy=deploy)
        self.act2 = nn.ReLU()

        if self.transform:
            self.reg_transform2_0 = feature_transform(feature_size, feature_size*4)

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
        # the number of all anchor points
        num_anchor_points = row * line

        if self.transform:
            self.backnone_transform0_0 = feature_transform(64, 128)
            self.backnone_transform1_0 = feature_transform(128, 256)
            self.backnone_transform2_0 = feature_transform(256, 512)
            self.backnone_transform3_0 = feature_transform(256, 512)

        # self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        # self.regression = RegressionModel(num_features_in=64, num_anchor_points=num_anchor_points, transform=self.transform)
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
            self.fpn_transform1_0 = feature_transform(256, 256*2)


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
            reg_features, regression = self.regression(features_fpn[1]) * 100  # 8x
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

    def loss_labels(self, outputs, targets, indices, num_points):
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

    def loss_points(self, outputs, targets, indices, num_points):

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

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}

        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))

        return losses

# create the P2PNet model
# try add perception loss to loss_sr
def build17_lite2_repvgg2(args, training, deploy=False):
    print("build P2PNetV17_lite2_repvgg2!")
    # print("build L1 Loss!")
    # treats persons as a single class
    num_classes = 1

    # add by wedream
    args.deploy = deploy
    # end

    backbone = build_backbone(args)
    model = P2PNet(backbone, args.row, args.line, args.deploy)
    if not training: 
        return model

    weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
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

    seed_everything(2024)

    from PIL import Image
    import torchvision.transforms as standard_transforms

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        # standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    backbone = Backbone_VGG('vgg16_bn_lite2', True, deploy=False)
    model1 = P2PNet(backbone, 2, 2, transform=False)
    # model.load_state_dict(torch.load('p2pnet.pth', map_location='cpu'), strict=True)
    model1.load_state_dict(torch.load('../weights/240402-lite2-repvgg2-02/best_mae.pth', map_location='cpu')['model'], strict=True)


    backbone = Backbone_VGG('vgg16_bn_lite2', True, deploy=True)
    model2 = P2PNet(backbone, 2, 2, transform=False, deploy=True)
    # model.load_state_dict(torch.load('p2pnet_deploy.pth', map_location='cpu'), strict=True)
    model2.load_state_dict(torch.load('../weights/240402-lite2-repvgg2-02/best_mae_deploy.pth', map_location='cpu'), strict=True)


    # set your image path here
    # img_path = r"D:\PycharmProject\CrowdCounting-P2PNet\datasets\ShanghaiTechFv2\part_A\test_data\images\IMG_1.jpg"
    img_dir = r"D:\PycharmProject\CrowdCounting-P2PNet\datasets\ShanghaiTechFv2\part_A\test_data\images"
    img_list = os.listdir(img_dir)

    for img_pth in img_list:
        img_path = os.path.join(img_dir, img_pth)

        print("img_path: {}".format(img_pth))

        # load the images
        img_raw = Image.open(img_path).convert('RGB')
        # round the size
        width, height = img_raw.size
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
        # pre-proccessing
        img = transform(img_raw)

        samples = torch.Tensor(img).unsqueeze(0)
        img = samples.to(device)

        # model1
        outputs = model1(img)
        # state_dict = model.state_dict()
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        predict_cnt = int((outputs_scores > 0.5).sum())
        print("predict_cnt1: {}".format(predict_cnt))  # [23.2093,  1.6285]

        # model2
        outputs = model2(img)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        predict_cnt = int((outputs_scores > 0.5).sum())
        print("predict_cnt2: {}".format(predict_cnt))  # [23.2093,  1.6285]

    #
    # # 安装 thop
    # # 基本用法
    # # import torch
    # from thop import profile  # 导入thop模块
    # model.eval()
    # # model = model.to(device)
    #
    # # input = torch.randn(1, 3, 384, 384)
    # input = torch.randn(1, 3, 1280, 640)
    # flops, params = profile(model, inputs=(input,))
    #
    # print('flops :%.3f' % (flops / 1000 ** 3), 'G')  # 打印计算量
    # print('params:%.3f' % (params / 1000 ** 2), 'M')  # 打印参数量

    # My MobileNetV2
    # flops :0.305 G
    # params:3.344 MB
