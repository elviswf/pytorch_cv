from model.base import *


class ResneXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, out_planes, stride=1, groups=32):
        super(ResneXtBottleneck, self).__init__()
        nb_filters = out_planes // self.expansion

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(in_planes, nb_filters, stride=stride)
        self.bn2 = nn.BatchNorm2d(nb_filters)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nb_filters, nb_filters, kernel_size=3, padding=1, groups=groups)
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


def _make_resnext_block(in_planes, out_planes, nb_sub_blocks, stride, block=ResneXtBottleneck, groups=32):
    layers = [block(in_planes, out_planes, stride=stride)]
    for _ in range(nb_sub_blocks - 1):
        layers.append(block(out_planes, out_planes, groups=groups))
    return nn.Sequential(*layers)


class ImagenetResNeXt(nn.Module):
    config = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
    }

    def __init__(self, depth=None, nb_classes=1000, blocks=None):
        super(ImagenetResNeXt, self).__init__()

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

        self.block1 = _make_resnext_block(64, 256, stride=1, nb_sub_blocks=blocks[0])
        self.block2 = _make_resnext_block(256, 512, stride=2, nb_sub_blocks=blocks[1])
        self.block3 = _make_resnext_block(512, 1024, stride=2, nb_sub_blocks=blocks[2])
        self.block4 = _make_resnext_block(1024, 2048, stride=2, nb_sub_blocks=blocks[3])

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


class CifarResNeXt(nn.Module):

    def __init__(self, depth, nb_classes):
        super(CifarResNeXt, self).__init__()
        if (depth - 2) % 9 != 0:
            raise Exception("depth should be 9n+2!")

        nb_sub_blocks = (depth - 2) // 9

        self.conv = conv3x3(3, 16)
        self.block1 = _make_resnext_block(16, 64, stride=1, nb_sub_blocks=nb_sub_blocks, groups=8)
        self.block2 = _make_resnext_block(64, 128, stride=2, nb_sub_blocks=nb_sub_blocks, groups=16)
        self.block3 = _make_resnext_block(128, 256, stride=2, nb_sub_blocks=nb_sub_blocks, groups=32)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(256, nb_classes)

        init_weights(self)

    def forward(self, x):
        x = self.conv(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = nn.AvgPool2d(x.size()[2:])(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def build_resnext(depth, nb_classes, mode='imagenet', blocks=None):
    if mode == "cifar":
        return CifarResNeXt(depth, nb_classes)
    elif mode == "imagenet":
        return ImagenetResNeXt(depth, nb_classes, blocks=blocks)
