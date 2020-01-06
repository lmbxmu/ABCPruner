import torch.nn as nn

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

cfgdepth = {
    1:64,
    2:64,
    3:128,
    4:128,
    5:256,
    6:256,
    7:256,
    8:512,
    9:512,
    10:512,
    11:512,
    12:512,
    13:512,
}

class VGG(nn.Module):
    def __init__(self, vgg_name, depth,num_classes=10):
        super(VGG, self).__init__()
        self.depth = depth
        self.features = self._make_layers(cfg[vgg_name])
        #self.classifier = nn.Linear(cfgdepth[depth], num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        depthnow = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if depthnow == self.depth:
                    break
                depthnow += 1
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

#honeysource: 1d向量，值限制在0-9
class BeeVGG(nn.Module):
    def __init__(self, vgg_name, honeysource, depth):
        super(BeeVGG, self).__init__()
        self.honeysource = honeysource
        self.depth = depth
        self.features = self._make_layers(cfg[vgg_name])
        #self.classifier = nn.Linear(int(cfgdepth[depth] * honeysource[len(honeysource)-1] / 10), 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out
        
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        index = 0
        depthnow = 0
        Mlayers = 0
        for x_index, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                Mlayers += 1
            else:
                if depthnow == self.depth:
                    break
                depthnow += 1
                x = int(x * self.honeysource[x_index - Mlayers] / 10)
                if x == 0:
                    x = 1
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



