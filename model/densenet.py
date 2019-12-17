import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

norm_mean, norm_var = 0.0, 1.0

class DenseBasicBlock(nn.Module):
    def __init__(self, inplanes, filters, index, expansion=1, growthRate=12, dropRate=0):
        super(DenseBasicBlock, self).__init__()
        planes = expansion * growthRate
        #print(inplanes, filters)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(filters, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        #print(x.size())
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        #print(out.size())
        #print(x.size())

        out = torch.cat((x, out), 1)
        #print(out.size())
        return out

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes, filters, index):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(filters, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #print(x.size())
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, depth=40, block=DenseBasicBlock,
        dropRate=0, num_classes=10, growthRate=12, compressionRate=2, filters=None, honey=None, indexes=None):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if 'DenseBasicBlock' in str(block) else (depth - 4) // 6
        
        if honey is None:
            self.honey = [10] * 36
        else:
            self.honey = honey

        for i in range(4):
            self.honey[8+i] = 10
            self.honey[20+i]=10
            self.honey[32+i]=10

        if filters == None:
            filters = []
            start = growthRate*2
            index = 0
            for i in range(3):
                index -= 1
                filter = 0
                for j in range(n+1):
                    if j != 0 :
                        filter += int(growthRate*self.honey[index]/10)
                    filters.append([start + filter])
                    index += 1
                start = (start + int(growthRate * self.honey[index-1]/10) * n) // compressionRate

            filters = [item for sub_list in filters for item in sub_list]	
            #print(filters)
           # print(len(filters))

            indexes = []
            for f in filters:
                indexes.append(np.arange(f))
            #print(indexes)

        self.growthRate = growthRate
        self.currentindex = 0
        self.dropRate = dropRate
        self.inplanes = growthRate * 2
        #self.inplanes = int(growthRate * 2 * self.honey[0] / 10)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3,padding=1,bias=False)



        self.dense1 = self._make_denseblock(block, n, filters[0:n], indexes[0:n])
        self.trans1 = self._make_transition(Transition, filters[n+1], filters[n], indexes[n])
        self.dense2 = self._make_denseblock(block, n, filters[n+1:2*n+1], indexes[n+1:2*n+1])
        self.trans2 = self._make_transition(Transition, filters[2*n+2], filters[2*n+1], indexes[2*n+1])
        self.dense3 = self._make_denseblock(block, n, filters[2*n+2:3*n+2], indexes[2*n+2:3*n+2])
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, filters, indexes):
        layers = []
        assert blocks == len(filters), 'Length of the filters parameter is not right.'
        assert blocks == len(indexes), 'Length of the indexes parameter is not right.'
        for i in range(blocks):
            # print("denseblock inplanes", filters[i])
            self.growthRate = int(12 * self.honey[self.currentindex] / 10)
            self.currentindex += 1
            self.inplanes = filters[i]
            layers.append(block(self.inplanes, filters=filters[i], index=indexes[i], growthRate=self.growthRate, dropRate=self.dropRate))
        self.inplanes += self.growthRate


        return nn.Sequential(*layers)


    def _make_transition(self, transition, compressionRate, filters, index):
        inplanes = self.inplanes
        outplanes = compressionRate
        self.inplanes = outplanes
        return transition(inplanes, outplanes, filters, index)


    def forward(self, x):
        x = self.conv1(x)

        x = self.dense1(x)
        x = self.trans1(x)
        #print("trans1 done!")
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def densenet(honey=None, **kwargs):
    return DenseNet(depth=40, block=DenseBasicBlock, compressionRate=1, honey=honey, **kwargs)

def test():
    honey = [1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9]
    model = densenet(honey = None)
    #ckpt = torch.load('../experience/densenet/baseline/checkpoint/densenet_40.pt', map_location='cpu')
    #model.load_state_dict(ckpt['state_dict'])
    #y = model(torch.randn(1, 3, 32, 32))
    print('Model.state_dict:')
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    #print(model)
    # for k, v in model.state_dict().items():
    #     print(k, v.size())

    # for name, module in model.named_modules():
    #     if isinstance(module, DenseBasicBlock):
    #         print(name)

#test()
