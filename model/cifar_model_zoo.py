from model.preact_resnet import build_preact_resent
from model.pyramid_net import PyramidNet
from model.dense_net import DenseNet
from model.resnet import build_bottleneck_resnet
from model.fantacy.random_swap import build_swap_preact_resent
from model.fantacy.pre_share import build_preact_resent_pre_share
from model.fantacy.post_share import build_preact_resent_post_share
from model.resnext import build_resnext
from model.resnext_cifar import CifarResNeXt


def resnet56(nb_classes):
    return build_bottleneck_resnet(56, nb_classes=nb_classes)


def resnet110(nb_classes):
    return build_bottleneck_resnet(110, nb_classes=nb_classes)


def pre_act_resnet56(nb_classes):
    return build_preact_resent(56, nb_classes=nb_classes, mode='cifar')


def swap_pre_act_resnet56(nb_classes):
    return build_swap_preact_resent(56, nb_classes=nb_classes, mode='cifar')


def pre_act_resnet56_pre_share1(nb_classes):
    return build_preact_resent_pre_share(56, nb_classes=nb_classes, mode='cifar', share=1)


def pre_act_resnet56_pre_share2(nb_classes):
    return build_preact_resent_pre_share(56, nb_classes=nb_classes, mode='cifar', share=2)


def pre_act_resnet56_pre_share3(nb_classes):
    return build_preact_resent_pre_share(56, nb_classes=nb_classes, mode='cifar', share=3)


def pre_act_resnet56_post_share1(nb_classes):
    return build_preact_resent_post_share(56, nb_classes=nb_classes, mode='cifar', share=1)


def pre_act_resnet56_post_share2(nb_classes):
    return build_preact_resent_post_share(56, nb_classes=nb_classes, mode='cifar', share=2)


def pre_act_resnet56_post_share3(nb_classes):
    return build_preact_resent_post_share(56, nb_classes=nb_classes, mode='cifar', share=3)


def pyramid_164_270_bottleneck(nb_classes):
    return PyramidNet(164, 270, num_classes=nb_classes, bottleneck=True)


def pyramid_272_200_bottleneck(nb_classes):
    return PyramidNet(272, 200, num_classes=nb_classes, bottleneck=True)


def pyramid_200_240_bottleneck(nb_classes):
    return PyramidNet(200, 240, num_classes=nb_classes, bottleneck=True)


def densenet_bc_40_12(nb_classes, drop_rate=0.):
    return DenseNet(40, 12, nb_classes=nb_classes, bottleneck=True, mode='cifar', drop_rate=drop_rate, compression=0.5)


def densenet_40_12(nb_classes, drop_rate=0.):
    return DenseNet(40, 12, nb_classes=nb_classes, bottleneck=False, mode='cifar', drop_rate=drop_rate, compression=1.0)


def densenet_bc_190_40(nb_classes, drop_rate=0.):
    return DenseNet(190, 40, nb_classes=nb_classes, bottleneck=True, mode='cifar', drop_rate=drop_rate, compression=0.5)


def resnext_56(nb_classes):
    return build_resnext(56, nb_classes=nb_classes, mode='cifar')


def resnet_164(nb_classes):
    return build_bottleneck_resnet(164, nb_classes, mode="cifar")


def resnext_29_8_64(nb_classes):
    return CifarResNeXt(cardinality=8, depth=29, nlabels=nb_classes, base_width=64)


def resnext_29_16_64(nb_classes):
    return CifarResNeXt(cardinality=16, depth=29, nlabels=nb_classes, base_width=64)


def pre_act_resnet_164(nb_classes):
    return build_preact_resent(164, nb_classes, mode="cifar")


def pre_act_resnet_110(nb_classes):
    return build_preact_resent(110, nb_classes, mode="cifar")


if __name__ == "__main__":
    import torch

    x = torch.randn(4, 3, 32, 32)
    model = densenet_bc_40_12(10, drop_rate=0.1)
    print(model)
    model = model.cuda()
    x = x.cuda()
    x_var = torch.autograd.Variable(x)
    y = model(x_var)
    z = model(x_var)
    print(y, z)
