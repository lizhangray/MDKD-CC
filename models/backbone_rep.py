# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import copy
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import numpy as np
import os

import models.vgg_rep as models

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        # print("features: {}".format(features))
        # print("len: {}".format(len(features)))
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
                # print("self.body1:{}".format(self.body1))
                # print("self.body2:{}".format(self.body2))
                # print("self.body3:{}".format(self.body3))
                # print("self.body4:{}".format(self.body4))
            elif name == 'vgg16_bn_lite':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
            elif name == 'vgg16_bn_lite2':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
            else:
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
            elif name == 'vgg16':
                self.body = nn.Sequential(*features[:30])  # 16x down-sample
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def forward(self, tensor_list):
        out = []

        if self.return_interm_layers:
            xs = tensor_list
            for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                out.append(xs)

        else:
            xs = self.body(tensor_list)
            out.append(xs)
        return out


class Backbone_VGG(BackboneBase_VGG):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, return_interm_layers: bool, deploy=False):
        if name == 'vgg16_bn':
            # backbone = models.vgg16_bn(pretrained=True)
            backbone = models.vgg16_bn(pretrained=False)

            # for name, parameters in backbone.named_parameters():
            #     print('name:', name)
            # for name, module in backbone._modules.items():
            #     print('name:', name)

        elif name == 'vgg16_bn_lite':
            # backbone = models.vgg16_bn(pretrained=True)
            # backbone = models.vgg16_bn_lite(pretrained=False)
            backbone = models.vgg16_bn_lite(pretrained=False, deploy=deploy)
            # print(backbone)
            # for name, parameters in backbone.named_parameters():
            #     print('name:', name)
            # for name, module in backbone._modules.items():
            #     print('name:', name)
        elif name == 'vgg16_bn_lite2':
            # backbone = models.vgg16_bn(pretrained=True)
            backbone = models.vgg16_bn_lite2(pretrained=False, deploy=deploy)
            # print(backbone)
            # for name, parameters in backbone.named_parameters():
            #     print('name:', name)
            # for name, module in backbone._modules.items():
            #     print('name:', name)

        elif name == 'vgg16':
            backbone = models.vgg16(pretrained=True)
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


def build_backbone(args):
    backbone = Backbone_VGG(args.backbone, True, deploy=args.deploy)
    return backbone

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
    # Backbone_VGG('vgg16', True)
    # Backbone_VGG('vgg16_bn_lite', True)
    # Backbone_VGG('vgg16_bn', True)

    seed_everything(2024)

    # img = torch.ones((8, 3, 128, 128))
    img = torch.rand((8, 3, 128, 128))
    model = Backbone_VGG('vgg16_bn_lite2', True)
    # print(model)
    # exit(0)
    # model = P2PNet(backbone, 2, 2, transform=False, deploy=False)
    out = model(img)
    state_dict = model.state_dict()
    torch.save(state_dict, 'vgg_rep.pth')
    print("out: {}".format(out[0][0][0][0][:10]))  # [0.7330, 4.7823, 3.2908, 2.9870]
    # # 模型转为部署模式保存
    # deploy_model = repvgg_model_convert(model, save_path='p2pnet_v17_kd2_deploy.pth', do_copy=False)
    deploy_model = repvgg_model_convert(model, save_path='vgg_deploy.pth')

    # img = torch.ones((8, 3, 128, 128))
    model = Backbone_VGG('vgg16_bn_lite2', True)
    # model = P2PNet(backbone, 2, 2, transform=False, deploy=False)
    model.load_state_dict(torch.load('vgg_rep.pth', map_location='cpu'), strict=True)
    out = model(img)
    state_dict = model.state_dict()
    # torch.save(state_dict, 'p2pnet_v17_kd2.pth')
    print("out: {}".format(out[0][0][0][0][:10]))  # [23.2093,  1.6285]

    # img = torch.ones((8, 3, 128, 128))
    model = Backbone_VGG('vgg16_bn_lite2', True, deploy=True)
    # model = P2PNet(backbone, 2, 2, transform=False, deploy=True)
    model.load_state_dict(torch.load('vgg_deploy.pth', map_location='cpu'), strict=True)
    out = model(img)
    print("out: {}".format(out[0][0][0][0][:10]))  # [23.2093,  1.6285]