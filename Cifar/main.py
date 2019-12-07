'''Train CIFAR-10 with PyTorch'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import os
import time
import utils.common as utils

from utils.options import args
from data import cifar100, cifar10
from importlib import import_module
from torch.optim.lr_scheduler import StepLR

device = torch.device(f"cuda:{args.gpus[0]}")
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'baselinelogger.log'))
loss_func = nn.CrossEntropyLoss()

# Training
def train(model, optimizer, trainLoader, args, epoch):

    model.train()
    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets)
        accurary.update(prec1[0], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Accurary {:.2f}%\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                    float(losses.avg), float(accurary.avg), cost_time
                )
            )
            start_time = current_time
#Testing
def test(model, testLoader):
    global best_acc
    model.eval()

    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tAccurary {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
        )
    return accurary.avg



def main():
    start_epoch = 0
    best_acc = 0.0

    # Data
    print('==> Preparing data..')
    if args.data_set == 'cifar10':
        loader = cifar10.Data(args)
        class_num = 10
    else:
        loader = cifar100.Data(args)
        class_num = 100
    
    #Model
    print('==> Building model..')
    if args.arch == 'vgg':
        model = import_module(f'model.{args.arch}').VGG(args.cfg, num_classes=class_num).to(device)
    elif args.arch == 'resnet':
        pass
    elif args.arch == 'googlenet':
        pass
    elif args.arch == 'densenet':
        pass

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

    if args.resume:
        print('=> Resuming from ckpt {}'.format(args.resume))
        ckpt = torch.load(args.resume, map_location=device)
        best_acc = ckpt['best_acc']
        start_epoch = ckpt['epoch']

        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        print('=> Continue from epoch {}...'.format(start_epoch))    

    if args.test_only:
        accuracy = test(model, loader.testLoader)
        print('=> Test Accurary: {:.2f}'.format(accuracy))
        return

    for epoch in range(start_epoch, args.num_epochs):
        train(model, optimizer, loader.trainLoader, args, epoch)
        scheduler.step()
        test_acc = test(model, loader.testLoader)

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
        
        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch+1, is_best)        
    
    logger.info('Best accurary: {:.3f}'.format(float(best_acc)))

if __name__ == '__main__':
    main()