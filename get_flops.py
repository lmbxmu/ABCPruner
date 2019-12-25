import torch
import argparse
import utils.common as utils
from importlib import import_module
from thop import profile

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument(
    '--input_image_size',
    type=int,
    default=32,
    help='The input_image_size')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_cifar',
    choices=('vgg_cifar','resnet_cifar','densenet','googlenet'),
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
    '--honey',
    type=int,
    nargs='+',
    default=None,
    help='The prune rate of CNN guided by best honey')
args = parser.parse_args()

device = torch.device("cpu")

print('==> Building model..')

if args.arch == 'vgg_cifar':
    orimodel = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
    model = import_module(f'model.{args.arch}').BeeVGG(args.cfg, honeysource=args.honey).to(device)
elif args.arch == 'resnet_cifar':
    orimodel = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    model = import_module(f'model.{args.arch}').resnet(args.cfg,honey=args.honey).to(device)
elif args.arch == 'googlenet':
    orimodel = import_module(f'model.{args.arch}').googlenet().to(device)
    model = import_module(f'model.{args.arch}').googlenet(honey=args.honey).to(device)
elif args.arch == 'densenet':
    orimodel = import_module(f'model.{args.arch}').densenet().to(device)
    model = import_module(f'model.{args.arch}').densenet(honey=args.honey).to(device)


input = torch.randn(1, 3, args.input_image_size, args.input_image_size)

print('--------------UnPrune Model--------------')
oriflops, oriparams = profile(orimodel, inputs=(input, ))
print('Params: %.2f'%(oriparams))
print('FLOPS: %.2f'%(oriflops))

print('--------------Prune Model--------------')
flops, params = profile(model, inputs=(input, ))
print('Params: %.2f'%(params))
print('FLOPS: %.2f'%(flops))

print('--------------Compress Rate--------------')
print('Params Compress Rate: %d/%d (%.2f%%)' % (params, oriparams, 100. * params / oriparams))
print('FLOPS Compress Rate: %d/%d (%.2f%%)' % (flops, oriflops, 100. * flops / oriflops))


