import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, in_planes, planes, stride=1, honey, index):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, int(planes * honey[index] / 10), kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(int(planes * honey[index] / 10))
		self.conv2 = nn.Conv2d(int(planes * honey[index] / 10), planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1, honey, index):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, int(planes * honeyrate / 10), kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(int())