from model.preact_resnet import build_preact_resent
from torchvision.models.densenet import densenet169, densenet121
from torchvision.models import resnet
from torchvision.models import inception
from model.resnext import build_resnext


def pre_act_resnet50(nb_classes=1000):
    return build_preact_resent(50, nb_classes=nb_classes, mode="imagenet")


def pre_act_resnet152(nb_classes=1000):
    return build_preact_resent(152, nb_classes=nb_classes, mode="imagenet")


def densenet_bc169(nb_classes=1000, pretrained=False):
    return densenet169(num_classes=nb_classes, pretrained=pretrained)


def densnet_bc121(nb_classes=1000, pretrained=False):
    return densenet121(pretrained=pretrained, num_classes=nb_classes)


def resnet50(nb_classes=1000, pretrained=False):
    return resnet.resnet50(pretrained=pretrained, num_classes=nb_classes)


def resnet101(nb_classes=1000, pretrained=False):
    return resnet.resnet101(pretrained=pretrained, num_classes=nb_classes)


def inception_v3(nb_classes=1000, pretrained=False):
    return inception.inception_v3(pretrained=pretrained, num_classes=nb_classes)


def resnext50(nb_classes=1000, pretrained=False):
    return build_resnext(50, nb_classes=nb_classes, mode='imagenet')
