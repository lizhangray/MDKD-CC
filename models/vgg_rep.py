# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Mostly copy-paste from torchvision references.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


model_paths = {
    # 'vgg16_bn': '/apdcephfs/private_changanwang/checkpoints/vgg16_bn-6c64b313.pth',
    'vgg16_bn': './weights/vgg16_bn-6c64b313.pth',
    'vgg16': '/apdcephfs/private_changanwang/checkpoints/vgg16-397923af.pth',

}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, sync=False, deploy=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d = RepBlock(in_channels, v, deploy=deploy)
            # conv2d = depthwise_separable_conv(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                if sync:
                    print('use sync backbone')
                    layers += [conv2d, nn.SyncBatchNorm(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# 重参数化技巧： (3*3) U (3*1+1*3+1*1) -> (3*3)
class RepBlock(nn.Module):
    # def __init__(self, in_channels, out_channels, act_type, deploy=False):
    def __init__(self, in_channels, out_channels, deploy=False):
        super(RepBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.activation = activation('lrelu')

        if deploy:
            # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(3, 3), stride=1,
                                         padding=1, dilation=1, groups=1, bias=True,
                                         padding_mode='zeros')
        else:
            self.rbr_3x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                            stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')
            self.rbr_3x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                            stride=1, padding=(1, 0), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                            stride=1, padding=(0, 1), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                            stride=1, padding=(0, 0), dilation=1, groups=1, padding_mode='zeros')

    def forward(self, inputs):
        # if self.deploy:
        #     return self.activation(self.rbr_reparam(inputs))
        # else:
        #     return self.activation( (self.rbr_3x3_branch(inputs)) +
        #                            (self.rbr_3x1_branch(inputs) + self.rbr_1x3_branch(inputs) + self.rbr_1x1_branch(inputs)) +
        #                            (inputs) )
        if self.deploy:
            return self.rbr_reparam(inputs)
        else:
            if self.out_channels == self.in_channels:
                return  (self.rbr_3x3_branch(inputs)) + \
                                       (self.rbr_3x1_branch(inputs) + self.rbr_1x3_branch(inputs) + self.rbr_1x1_branch(inputs)) + \
                                       (inputs)
            else:
                return (self.rbr_3x3_branch(inputs)) + \
                       (self.rbr_3x1_branch(inputs) + self.rbr_1x3_branch(inputs) + self.rbr_1x1_branch(inputs))


    def switch_to_deploy(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                                     stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_3x3_branch')
        self.__delattr__('rbr_3x1_branch')
        self.__delattr__('rbr_1x3_branch')
        self.__delattr__('rbr_1x1_branch')
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        # 3x3 branch
        kernel_3x3, bias_3x3 = self.rbr_3x3_branch.weight.data, self.rbr_3x3_branch.bias.data
        # 1x1 1x3 3x1 branch
        kernel_1x1_1x3_3x1_fuse, bias_1x1_1x3_3x1_fuse = self._fuse_1x1_1x3_3x1_branch(self.rbr_1x1_branch,
                                                                                       self.rbr_1x3_branch,
                                                                                  self.rbr_3x1_branch)

        if self.out_channels == self.in_channels:
            # identity branch
            device = kernel_1x1_1x3_3x1_fuse.device  # just for getting the device
            kernel_identity = torch.zeros(self.out_channels, self.in_channels, 3, 3, device=device)
            for i in range(self.out_channels):
                kernel_identity[i, i, 1, 1] = 1.0

            return kernel_3x3 + kernel_1x1_1x3_3x1_fuse + kernel_identity, \
                   bias_3x3 + bias_1x1_1x3_3x1_fuse

        else:
            return kernel_3x3 + kernel_1x1_1x3_3x1_fuse, \
                   bias_3x3 + bias_1x1_1x3_3x1_fuse


    def _fuse_1x1_1x3_3x1_branch(self, conv1, conv2, conv3):
        weight = F.pad(conv1.weight.data, (1, 1, 1, 1)) + F.pad(conv2.weight.data, (0, 0, 1, 1)) + F.pad(
            conv3.weight.data, (1, 1, 0, 0))
        bias = conv1.bias.data + conv2.bias.data + conv3.bias.data
        return weight, bias


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
         'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_lite': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'],
    'D_lite2': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, sync=False, deploy=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    # model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, sync=sync), **kwargs)
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, sync=sync, deploy=deploy), **kwargs)
    if pretrained:
        state_dict = torch.load(model_paths[arch])
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, sync=False, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, sync=sync, **kwargs)


def vgg16_bn_lite(pretrained=False, progress=True, sync=False, deploy=False, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D_lite', True, pretrained, progress, sync=sync, deploy=deploy, **kwargs)

def vgg16_bn_lite2(pretrained=False, progress=True, sync=False, deploy=False, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # return _vgg('vgg16_bn', 'D_lite2', True, pretrained, progress, sync=sync, **kwargs)
    return _vgg('vgg16_bn', 'D_lite2', True, pretrained, progress, sync=sync, deploy=deploy, **kwargs)

def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)


# import torch.nn as nn
# 深度可分离卷积
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
