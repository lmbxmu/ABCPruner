import torch
import torch.nn as nn
import argparse
import utils.common as utils
from importlib import import_module
from thop import profile

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')


parser.add_argument(
    '--arch',
    type=str,
    default='vgg_cifar',
    choices=('vgg_cifar','resnet_cifar','vgg','resnet','densenet','googlenet','vgglayerwise'),
    help='The architecture to prune')
parser.add_argument(
    '--data_set',
    type=str,
    default='cifar10',)
parser.add_argument(
    '--cfg',
    type=str,
    default='resnet56'
)
parser.add_argument(
    '--gpus',
    type=int,
    nargs='+',
    default=[0],
    help='Select gpu_id to use. default:[0]',
)

parser.add_argument(
    '--depth',
    type=int,
    default=None,
    )

parser.add_argument(
    '--honey',
    type=str,
    default=None,
    help='The prune rate of CNN guided by best honey')
args = parser.parse_args()
honey = list(map(int,args.honey.split(', ')))



device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

print('==> Building model..')

if args.arch == 'vgg_cifar':
    orimodel = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
    model = import_module(f'model.{args.arch}').BeeVGG(args.cfg, honeysource=honey).to(device)
elif args.arch == 'resnet_cifar':
    orimodel = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    model = import_module(f'model.{args.arch}').resnet(args.cfg,honey=honey).to(device)
elif args.arch == 'vgg':
    orimodel = import_module(f'model.{args.arch}').VGG(num_classes=1000).to(device)
    model = import_module(f'model.{args.arch}').BeeVGG(honeysource=honey, num_classes = 1000).to(device)
elif args.arch == 'resnet':
    orimodel = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    model = import_module(f'model.{args.arch}').resnet(args.cfg,honey=honey).to(device)
elif args.arch == 'googlenet':
    orimodel = import_module(f'model.{args.arch}').googlenet().to(device)
    model = import_module(f'model.{args.arch}').googlenet(honey=honey).to(device)
elif args.arch == 'densenet':
    orimodel = import_module(f'model.{args.arch}').densenet().to(device)
    model = import_module(f'model.{args.arch}').densenet(honey=honey).to(device)
elif args.arch == 'vgglayerwise':
    orimodel = import_module(f'model.{args.arch}').VGG(args.cfg, depth = args.depth).to(device)
    model = import_module(f'model.{args.arch}').BeeVGG(args.cfg, honeysource=honey, depth = args.depth).to(device)

if args.data_set == 'cifar10':
    input_image_size = 32
elif args.data_set == 'imagenet':
    input_image_size = 224

input = torch.randn(1, 3, input_image_size, input_image_size).to(device)

orichannel = 0
channel = 0


oriflops, oriparams = profile(orimodel, inputs=(input, ))
flops, params = profile(model, inputs=(input, ))


for name, module in orimodel.named_modules():

        if isinstance(module, nn.Conv2d):
            orichannel += orimodel.state_dict()[name + '.weight'].size(0)
            #print(orimodel.state_dict()[name + '.weight'].size(0))

for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):
            channel += model.state_dict()[name + '.weight'].size(0)
            #print(model.state_dict()[name + '.weight'].size(0))

print('--------------UnPrune Model--------------')
print('Channels: %d'%(orichannel))
print('Params: %.2f M '%(oriparams/1000000))
print('FLOPS: %.2f M '%(oriflops/1000000))

print('--------------Prune Model--------------')
print('Channels:%d'%(channel))
print('Params: %.2f M'%(params/1000000))
print('FLOPS: %.2f M'%(flops/1000000))


print('--------------Compress Rate--------------')
print('Channels Prune Rate: %d/%d (%.2f%%)' % (channel, orichannel, 100. * (orichannel - channel) / orichannel))
print('Params Compress Rate: %.2f M/%.2f M(%.2f%%)' % (params/1000000, oriparams/1000000, 100. * (oriparams- params) / oriparams))
print('FLOPS Compress Rate: %.2f M/%.2f M(%.2f%%)' % (flops/1000000, oriflops/1000000, 100. * (oriflops- flops) / oriflops))

