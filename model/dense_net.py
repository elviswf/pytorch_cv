from model.base import *


class DenseLayer(nn.Sequential):
    multiplier = 4

    def __init__(self, in_planes, growth_rate, drop_rate, bottleneck):
        super(DenseLayer, self).__init__()
        if bottleneck:
            inter_planes = growth_rate * self.multiplier
            self.add_module('bn1', nn.BatchNorm2d(in_planes))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', conv1x1(in_planes, inter_planes))
            self.add_module('bn2', nn.BatchNorm2d(inter_planes))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv2', conv3x3(inter_planes, growth_rate))
        else:
            self.add_module('bn1', nn.BatchNorm2d(in_planes))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', conv3x3(in_planes, growth_rate))

        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout(p=drop_rate))

    def forward(self, x):
        feature = super(DenseLayer, self).forward(x)
        return torch.cat([x, feature], 1)


class DenseBlock(nn.Sequential):

    def __init__(self, nb_layers, in_planes, growth_rate, bottleneck=False, drop_rate=0.):
        super(DenseBlock, self).__init__()

        for i in range(nb_layers):
            layer = DenseLayer(in_planes + i * growth_rate, growth_rate, bottleneck=bottleneck, drop_rate=drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class Transition(nn.Sequential):

    def __init__(self, in_planes, out_planes, drop_rate):
        super(Transition, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_planes))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', conv1x1(in_planes, out_planes))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout(p=drop_rate))
        self.add_module('avg_pool', nn.AvgPool2d(kernel_size=2, stride=2))


def cifar_input(model, depth, growth_rate, bottleneck):
    if (depth - 4) % 3 != 0:
        raise Exception("depth should be 3*n + 4!")
    nb_layers = (depth - 4) // 3
    init_planes = 16
    if bottleneck:
        nb_layers = nb_layers // 2
        init_planes = 2 * growth_rate
    model.add_module("conv0", conv3x3(3, init_planes))

    return init_planes, [nb_layers] * 3


def imagenet_input(model, depth, growth_rate, bottleneck):
    config_bc = {
        121: (6, 12, 24, 16),
        161: (6, 12, 36, 24),
        169: (6, 12, 32, 32),
        201: (6, 12, 48, 32)
    }

    blocks = config_bc.get(depth, None)
    if blocks is None:
        raise Exception("dont support block config for densenet-%d." % depth)
    if not bottleneck:
        blocks *= 2
    in_planes = 2 * growth_rate
    model.add_module("conv0", nn.Conv2d(3, in_planes, kernel_size=7, stride=2, padding=3, bias=False))
    model.add_module("bn0", nn.BatchNorm2d(in_planes))
    model.add_module("relu0", nn.ReLU(inplace=True))
    model.add_module("max_pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    return in_planes, blocks


class DenseNet(nn.Module):

    def __init__(self, depth, growth_rate, nb_classes, bottleneck=False, compression=1., drop_rate=0., mode='cifar'):
        super(DenseNet, self).__init__()
        self.feature = nn.Sequential()
        if isinstance(mode, str):
            input_config = {"cifar": cifar_input, "imagenet": imagenet_input}
            func = input_config.get(mode, None)
            if func is None:
                raise Exception("don't have input configuration!")
            in_planes, blocks = func(self.feature, depth, growth_rate, bottleneck)
        else:
            in_planes, blocks = mode(self.feature, **locals())

        for n, nb_layers in enumerate(blocks):
            self.feature.add_module("dense_block_%d" % n,
                                    DenseBlock(nb_layers, in_planes, growth_rate, bottleneck, drop_rate))
            in_planes += nb_layers * growth_rate
            if n != len(blocks) - 1:
                out_planes = int(in_planes * compression)
                self.feature.add_module("trans_%d" % n, Transition(in_planes, out_planes, drop_rate))
                in_planes = out_planes
        self.feature.add_module('bn', nn.BatchNorm2d(in_planes))
        self.feature.add_module('relu', nn.ReLU(inplace=True))
        self.fc = nn.Linear(in_planes, nb_classes)

        init_weights(self)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(x.size()[2:])(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # TODO: add command interface


def unit_test():
    model = DenseNet(201, 32, nb_classes=1000, bottleneck=True, compression=0.5, mode="imagenet")
    print("parameter number: ", get_n_params(model))
    # print(model)


if __name__ == "__main__":
    unit_test()
