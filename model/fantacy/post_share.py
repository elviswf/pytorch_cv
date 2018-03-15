from model.base import *


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(Bottleneck, self).__init__()
        nb_filters = out_planes // self.expansion

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(in_planes, nb_filters, stride=stride)
        self.bn2 = nn.BatchNorm2d(nb_filters)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(nb_filters, nb_filters)
        self.bn3 = nn.BatchNorm2d(nb_filters)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(nb_filters, out_planes)

        if in_planes != out_planes or stride != 1:
            self.shortcut = conv1x1(in_planes, out_planes, stride=stride)
        else:
            self.shortcut = None

    def forward(self, input):
        x = self.bn1(input)
        x = self.relu1(x)

        y = self.conv1(x)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.conv2(y)
        y = self.bn3(y)
        y = self.relu3(y)
        y = self.conv3(y)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = input

        out = shortcut + y
        return out


def _make_resnet_block(in_planes, out_planes, nb_sub_blocks, stride, block=Bottleneck):
    layers = [block(in_planes, out_planes, stride=stride)]
    for _ in range(nb_sub_blocks - 1):
        layers.append(block(out_planes, out_planes))
    return nn.Sequential(*layers)


class CifarPreActResNetPostShare(nn.Module):

        def __init__(self, depth, nb_classes, separate_end=0):
            super(CifarPreActResNetPostShare, self).__init__()
            if (depth - 2) % 9 != 0:
                raise Exception("depth should be 9n+2!")

            nb_sub_blocks = (depth - 2) // 9

            self.separate_end = separate_end
            self.conv = conv3x3(3, 16)
            self.conv2 = conv3x3(3, 16)

            self.block1 = _make_resnet_block(16, 64, stride=1, nb_sub_blocks=nb_sub_blocks)
            self.block2 = _make_resnet_block(64, 128, stride=2, nb_sub_blocks=nb_sub_blocks)
            self.block3 = _make_resnet_block(128, 256, stride=2, nb_sub_blocks=nb_sub_blocks)

            if separate_end >= 3:
                self.clone3 = _make_resnet_block(128, 256, stride=2, nb_sub_blocks=nb_sub_blocks)
            if separate_end >= 2:
                self.clone2 = _make_resnet_block(64, 128, stride=2, nb_sub_blocks=nb_sub_blocks)
            if separate_end >= 1:
                self.clone1 = _make_resnet_block(16, 64, stride=2, nb_sub_blocks=nb_sub_blocks)

            self.bn = nn.BatchNorm2d(256)
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(256, nb_classes)

            init_weights(self)

        def forward(self, x):
            x0 = self.conv(x)
            y0 = self.conv2(x)

            x1 = self.block1(x0)
            x2 = self.block2(x1)
            x3 = self.block3(x2)

            x = self.bn(x3)
            x = self.relu(x)
            x = nn.AvgPool2d(x.size()[2:])(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            if self.separate_end >= 1:
                y1 = self.clone1(y0)
            else:
                y1 = self.block1(y0)

            if self.separate_end >= 2:
                y2 = self.clone2(y1)
            else:
                y2 = self.block2(y1)

            if self.separate_end >= 3:
                y3 = self.clone3(y2)
            else:
                y3 = self.block3(y2)

            y = self.bn(y3)
            y = self.relu(y)
            y = nn.AvgPool2d(y.size()[2:])(y)
            y = y.view(y.size(0), -1)
            y = self.fc(y)

            return [x, y]


class ImagenetPreActResNet(nn.Module):

    config = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
    }

    def __init__(self, depth=None, nb_classes=1000, blocks=None):
        super(ImagenetPreActResNet, self).__init__()

        if blocks is None:
            blocks = self.config.get(depth, None)
            if blocks is None:
                raise Exception("can't determine the blocks configuration! parameter blocks is required!")
        else:
            if len(blocks) != 4:
                raise Exception("blocks should be a list of 4 ints!")

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = _make_resnet_block(64, 256, stride=1, nb_sub_blocks=blocks[0])
        self.block2 = _make_resnet_block(256, 512, stride=2, nb_sub_blocks=blocks[1])
        self.block3 = _make_resnet_block(512, 1024, stride=2, nb_sub_blocks=blocks[2])
        self.block4 = _make_resnet_block(1024, 2048, stride=2, nb_sub_blocks=blocks[3])

        self.bn2 = nn.BatchNorm2d(2048)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2048, nb_classes)

        init_weights(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.max_pool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = nn.AvgPool2d(x.size()[2:])(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def build_preact_resent_post_share(depth, nb_classes, mode='cifar', blocks=None, share=0):
    if mode == "cifar":
        return CifarPreActResNetPostShare(depth, nb_classes, separate_end=share)
    elif mode == "imagenet":
        return ImagenetPreActResNet(depth, nb_classes, blocks=blocks)


def main():
    import argparse
    # TODO: add command interface


def unit_test():
    model = build_preact_resent_post_share(110, 10, mode='cifar', share=2)
    print("parameter number: ", get_n_params(model))


if __name__ == "__main__":
    unit_test()