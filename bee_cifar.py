# The ABC algorithm is modified based on https://www.cnblogs.com/ybl20000418/p/11366576.html （In Chinese）
import torch
import torch.nn as nn
import torch.optim as optim
from model.googlenet import Inception
from utils.options import args
import utils.common as utils

import os
import time
import copy
import sys
import random
import numpy as np
import heapq
from data import cifar10, cifar100, imagenet
from importlib import import_module

checkpoint = utils.checkpoint(args)
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

conv_num_cfg = {
    'vgg16': 13,
    'resnet56' : 27,
    'resnet110' : 54,
    'googlenet' : 9,
    'densenet':36,
    }
food_dimension = conv_num_cfg[args.cfg]

# Data
print('==> Loading Data..')
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
elif args.data_set == 'cifar100':
    loader = cifar100.Data(args)
else:
    loader = imagenet.Data(args)


# Model
print('==> Loading Model..')
if args.arch == 'vgg_cifar':
    origin_model = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
elif args.arch == 'resnet_cifar':
    origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
elif args.arch == 'googlenet':
    origin_model = import_module(f'model.{args.arch}').googlenet().to(device)
elif args.arch == 'densenet':
    origin_model = import_module(f'model.{args.arch}').densenet().to(device)

if args.honey_model is None or not os.path.exists(args.honey_model):
    raise ('Honey_model path should be exist!')


ckpt = torch.load(args.honey_model, map_location=device)
origin_model.load_state_dict(ckpt['state_dict'])
oristate_dict = origin_model.state_dict()


#Define BeeGroup 
class BeeGroup():
    """docstring for BeeGroup"""
    def __init__(self):
        super(BeeGroup, self).__init__() 
        self.code = [] #size : num of conv layers  value:{1,2,3,4,5,6,7,8,9,10}
        self.fitness = 0
        self.rfitness = 0 
        self.trail = 0

#Initilize global element
best_honey = BeeGroup()
NectraSource = []
EmployedBee = []
OnLooker = []
best_honey_state = {}



#load pre-train params
def load_vgg_honey_model(model, random_rule):
    #print(ckpt['state_dict'])
    global oristate_dict
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                select_num = currentfilter_num
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num-1), select_num)
                    select_index.sort()
                else:
                    l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    select_index.sort()
                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            else:
                state_dict[name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)

def load_dense_honey_model(model, random_rule):

    global oristate_dict
    

    state_dict = model.state_dict()

    conv_weight = []
    conv_trans_weight = []
    bn_weight = []
    bn_bias = []

    for i in range(3):
        for j in range(12):
            conv1_weight_name = 'dense%d.%d.conv1.weight' % (i + 1, j)
            conv_weight.append(conv1_weight_name)

            bn1_weight_name = 'dense%d.%d.bn1.weight' % (i + 1, j)
            bn_weight.append(bn1_weight_name)

            bn1_bias_name = 'dense%d.%d.bn1.bias' %(i+1,j)
            bn_bias.append(bn1_bias_name)

    for i in range(2):
        conv1_weight_name = 'trans%d.conv1.weight' % (i + 1)
        conv_weight.append(conv1_weight_name)
        conv_trans_weight.append(conv1_weight_name)

        bn_weight_name = 'trans%d.bn1.weight' % (i + 1)
        bn_weight.append(bn_weight_name)

        bn_bias_name = 'trans%d.bn1.bias' % (i + 1)
        bn_bias.append(bn_bias_name)
    
    bn_weight.append('bn.weight')
    bn_bias.append('bn.bias')


    
    for k in range(len(conv_weight)):
        conv_weight_name = conv_weight[k]
        oriweight = oristate_dict[conv_weight_name]
        curweight = state_dict[conv_weight_name]
        orifilter_num = oriweight.size(1)
        currentfilter_num = curweight.size(1)
        select_num = currentfilter_num
        #print(orifilter_num)
        #print(currentfilter_num)

        if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
            if random_rule == 'random_pretrain':
                select_index = random.sample(range(0, orifilter_num-1), select_num)
                select_index.sort()
            else:
                l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                select_index.sort()

            for i in range(curweight.size(0)):
                for index_j, j in enumerate(select_index):
                    state_dict[conv_weight_name][i][index_j] = \
                            oristate_dict[conv_weight_name][i][j]


    for k in range(len(bn_weight)):

        bn_weight_name = bn_weight[k]
        bn_bias_name = bn_bias[k]
        bn_bias.append(bn_bias_name)
        bn_weight.append(bn_weight_name)
        oriweight = oristate_dict[bn_weight_name]
        curweight = state_dict[bn_weight_name]

        orifilter_num = oriweight.size(0)
        currentfilter_num = curweight.size(0)
        select_num = currentfilter_num

        if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
            if random_rule == 'random_pretrain':
                select_index = random.sample(range(0, orifilter_num-1), select_num)
                select_index.sort()
            else:
                l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                select_index.sort()

            for index_j, j in enumerate(select_index):
                state_dict[bn_weight_name][index_j] = \
                        oristate_dict[bn_weight_name][j]
                state_dict[bn_bias_name][index_j] = \
                        oristate_dict[bn_bias_name][j]

    oriweight = oristate_dict['fc.weight']
    curweight = state_dict['fc.weight']
    orifilter_num = oriweight.size(1)
    currentfilter_num = curweight.size(1)
    select_num = currentfilter_num

    if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
        if random_rule == 'random_pretrain':
            select_index = random.sample(range(0, orifilter_num-1), select_num)
            select_index.sort()
        else:
            l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
            select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
            select_index.sort()

        for i in range(curweight.size(0)): 
            for index_j, j in enumerate(select_index):
                state_dict['fc.weight'][i][index_j] = \
                        oristate_dict['fc.weight'][i][j]



    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.BatchNorm2d):
            bn_weight_name = name + '.weight'
            bn_bias_name = name + '.bias'
            if bn_weight_name not in bn_weight and bn_bias_name not in bn_bias:
                state_dict[bn_weight_name] = oristate_dict[bn_weight_name]
                state_dict[bn_bias_name] = oristate_dict[bn_bias_name]

    model.load_state_dict(state_dict)




def load_google_honey_model(model, random_rule):
    global oristate_dict
    state_dict = model.state_dict()
        
    last_select_index = None
    all_honey_conv_name = []
    all_honey_bn_name = []

    for name, module in model.named_modules():

        if isinstance(module, Inception):

            honey_filter_channel_index = ['.branch5x5.3']  # the index of sketch filter and channel weight
            honey_channel_index = ['.branch3x3.3', '.branch5x5.6']  # the index of sketch channel weight
            honey_filter_index = ['.branch3x3.0', '.branch5x5.0']  # the index of sketch filter weight
            honey_bn_index = ['.branch3x3.1', '.branch5x5.1', '.branch5x5.4'] #the index of sketch bn weight
            
            for bn_index in honey_bn_index:
                all_honey_bn_name.append(name + bn_index)

            for weight_index in honey_filter_channel_index:
                last_select_index = None
                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]

                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    #print(state_dict[conv_name].size())
                    #print(oristate_dict[conv_name].size())

                


                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)


                select_index_1 = copy.deepcopy(select_index)


                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                for index_i, i in enumerate(select_index):
                    for index_j, j in enumerate(select_index_1):
                            state_dict[conv_name][index_i][index_j] = \
                                oristate_dict[conv_name][i][j]



            for weight_index in honey_channel_index:

                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]
                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                #print(state_dict[conv_name].size())
                #print(oristate_dict[conv_name].size())


                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()


                    for i in range(state_dict[conv_name].size(0)):
                        for index_j, j in enumerate(select_index):
                            state_dict[conv_name][i][index_j] = \
                                oristate_dict[conv_name][i][j]


            for weight_index in honey_filter_index:

                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)
                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    for index_i, i in enumerate(select_index):
                            state_dict[conv_name][index_i] = \
                                oristate_dict[conv_name][i]


    for name, module in model.named_modules(): #Reassign non sketch weights to the new network

        if isinstance(module, nn.Conv2d):

            if name not in all_honey_conv_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']

        elif isinstance(module, nn.BatchNorm2d):

            if name not in all_honey_bn_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']
                state_dict[name + '.running_mean'] = oristate_dict[name + '.running_mean']
                state_dict[name + '.running_var'] = oristate_dict[name + '.running_var']

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


def load_resnet_honey_model(model, random_rule):

    cfg = { 
           'resnet56': [9,9,9],
           'resnet110': [18,18,18],
           }

    global oristate_dict
    state_dict = model.state_dict()
        
    current_cfg = cfg[args.cfg]
    last_select_index = None

    all_honey_conv_weight = []

    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):
                conv_name = layer_name + str(k) + '.conv' + str(l+1)
                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)
                #logger.info('weight_num {}'.format(conv_weight_name))
                #logger.info('orifilter_num {}\tcurrentnum {}\n'.format(orifilter_num,currentfilter_num))
                #logger.info('orifilter  {}\tcurrent {}\n'.format(oristate_dict[conv_weight_name].size(),state_dict[conv_weight_name].size()))

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()
                    if last_select_index is not None:
                        logger.info('last_select_index'.format(last_select_index))
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]  

                    last_select_index = select_index
                    #logger.info('last_select_index{}'.format(last_select_index)) 

                elif last_select_index != None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    #for param_tensor in state_dict:
        #logger.info('param_tensor {}\tType {}\n'.format(param_tensor,state_dict[param_tensor].size()))
    #for param_tensor in model.state_dict():
        #logger.info('param_tensor {}\tType {}\n'.format(param_tensor,model.state_dict()[param_tensor].size()))
 

    model.load_state_dict(state_dict)


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

#Testinga
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

#Calculate fitness of a honey source
def calculationFitness(honey, train_loader, args):
    global best_honey
    global best_honey_state

    if args.arch == 'vgg_cifar':
        model = import_module(f'model.{args.arch}').BeeVGG(args.cfg,honeysource=honey).to(device)
        load_vgg_honey_model(model, args.random_rule)
    elif args.arch == 'resnet_cifar':
        model = import_module(f'model.{args.arch}').resnet(args.cfg,honey=honey).to(device)
        load_resnet_honey_model(model, args.random_rule)
    elif args.arch == 'googlenet':
        model = import_module(f'model.{args.arch}').googlenet(honey=honey).to(device)
        load_google_honey_model(model, args.random_rule)
    elif args.arch == 'densenet':
        model = import_module(f'model.{args.arch}').densenet(honey=honey).to(device)
        load_dense_honey_model(model, args.random_rule)

    fit_accurary = utils.AverageMeter()
    train_accurary = utils.AverageMeter()



    #start_time = time.time()
    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #test(model, loader.testLoader)

    model.train()
    for epoch in range(args.calfitness_epoch):
        for batch, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            #print("ok")
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()

            prec1 = utils.accuracy(output, targets)
            train_accurary.update(prec1[0], inputs.size(0))

    #test(model, loader.testLoader)

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader.testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = utils.accuracy(outputs, targets)
            fit_accurary.update(predicted[0], inputs.size(0))


    #current_time = time.time()
    '''
    logger.info(
            'Honey Source fintness {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(accurary.avg), (current_time - start_time))
        )
    '''
    if fit_accurary.avg > best_honey.fitness:
        best_honey_state = copy.deepcopy(model.module.state_dict() if len(args.gpus) > 1 else model.state_dict())
        best_honey.code = copy.deepcopy(honey)
        best_honey.fitness = fit_accurary.avg

    return fit_accurary.avg


#Initilize Bee-Pruning
def initilize():
    print('==> Initilizing Honey_model..')
    global best_honey, NectraSource, EmployedBee, OnLooker

    for i in range(args.food_number):
        NectraSource.append(copy.deepcopy(BeeGroup()))
        EmployedBee.append(copy.deepcopy(BeeGroup()))
        OnLooker.append(copy.deepcopy(BeeGroup()))
        for j in range(food_dimension):
            NectraSource[i].code.append(copy.deepcopy(random.randint(1,args.max_preserve)))

        #initilize honey souce
        NectraSource[i].fitness = calculationFitness(NectraSource[i].code, loader.trainLoader, args)
        NectraSource[i].rfitness = 0
        NectraSource[i].trail = 0

        #initilize employed bee  
        EmployedBee[i].code = copy.deepcopy(NectraSource[i].code)
        EmployedBee[i].fitness=NectraSource[i].fitness 
        EmployedBee[i].rfitness=NectraSource[i].rfitness 
        EmployedBee[i].trail=NectraSource[i].trail

        #initilize onlooker 
        OnLooker[i].code = copy.deepcopy(NectraSource[i].code)
        OnLooker[i].fitness=NectraSource[i].fitness 
        OnLooker[i].rfitness=NectraSource[i].rfitness 
        OnLooker[i].trail=NectraSource[i].trail

    #initilize best honey
    best_honey.code = copy.deepcopy(NectraSource[0].code)
    best_honey.fitness = NectraSource[0].fitness
    best_honey.rfitness = NectraSource[0].rfitness
    best_honey.trail = NectraSource[0].trail

#Send employed bees to find better honey source
def sendEmployedBees():
    global NectraSource, EmployedBee
    for i in range(args.food_number):
        
        while 1:
            k = random.randint(0, args.food_number-1)
            if k != i:
                break

        EmployedBee[i].code = copy.deepcopy(NectraSource[i].code)

        param2change = np.random.randint(0, food_dimension-1, args.honeychange_num)
        R = np.random.uniform(-1, 1, args.honeychange_num)
        for j in range(args.honeychange_num):
            EmployedBee[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]]+ R[j]*(NectraSource[i].code[param2change[j]]-NectraSource[k].code[param2change[j]]))
            if EmployedBee[i].code[param2change[j]] < 1:
                EmployedBee[i].code[param2change[j]] = 1
            if EmployedBee[i].code[param2change[j]] > args.max_preserve:
                EmployedBee[i].code[param2change[j]] = args.max_preserve

        EmployedBee[i].fitness = calculationFitness(EmployedBee[i].code, loader.trainLoader, args)

        if EmployedBee[i].fitness > NectraSource[i].fitness:                
            NectraSource[i].code = copy.deepcopy(EmployedBee[i].code)              
            NectraSource[i].trail = 0  
            NectraSource[i].fitness = EmployedBee[i].fitness 
            
        else:          
            NectraSource[i].trail = NectraSource[i].trail + 1

#Calculate whether a Onlooker to update a honey source
def calculateProbabilities():
    global NectraSource
    
    maxfit = NectraSource[0].fitness

    for i in range(1, args.food_number):
        if NectraSource[i].fitness > maxfit:
            maxfit = NectraSource[i].fitness

    for i in range(args.food_number):
        NectraSource[i].rfitness = (0.9 * (NectraSource[i].fitness / maxfit)) + 0.1

#Send Onlooker bees to find better honey source
def sendOnlookerBees():
    global NectraSource, EmployedBee, OnLooker
    i = 0
    t = 0
    while t < args.food_number:
        R_choosed = random.uniform(0,1)
        if(R_choosed < NectraSource[i].rfitness):
            t += 1

            while 1:
                k = random.randint(0, args.food_number-1)
                if k != i:
                    break
            OnLooker[i].code = copy.deepcopy(NectraSource[i].code)

            param2change = np.random.randint(0, food_dimension-1, args.honeychange_num)
            R = np.random.uniform(-1, 1, args.honeychange_num)
            for j in range(args.honeychange_num):
                OnLooker[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]]+ R[j]*(NectraSource[i].code[param2change[j]]-NectraSource[k].code[param2change[j]]))
                if OnLooker[i].code[param2change[j]] < 1:
                    OnLooker[i].code[param2change[j]] = 1
                if OnLooker[i].code[param2change[j]] > args.max_preserve:
                    OnLooker[i].code[param2change[j]] = args.max_preserve

            OnLooker[i].fitness = calculationFitness(OnLooker[i].code, loader.trainLoader, args)

            if OnLooker[i].fitness > NectraSource[i].fitness:                
                NectraSource[i].code = copy.deepcopy(OnLooker[i].code)              
                NectraSource[i].trail = 0  
                NectraSource[i].fitness = OnLooker[i].fitness 
            else:          
                NectraSource[i].trail = NectraSource[i].trail + 1
        i += 1
        if i == args.food_number:
            i = 0

#If a honey source has not been update for args.food_limiet times, send a scout bee to regenerate it
def sendScoutBees():
    global  NectraSource, EmployedBee, OnLooker
    maxtrailindex = 0
    for i in range(args.food_number):
        if NectraSource[i].trail > NectraSource[maxtrailindex].trail:
            maxtrailindex = i
    if NectraSource[maxtrailindex].trail >= args.food_limit:
        for j in range(food_dimension):
            R = random.uniform(0,1)
            NectraSource[maxtrailindex].code[j] = int(R * args.max_preserve)
            if NectraSource[maxtrailindex].code[j] == 0:
                NectraSource[maxtrailindex].code[j] += 1
        NectraSource[maxtrailindex].trail = 0
        NectraSource[maxtrailindex].fitness = calculationFitness(NectraSource[maxtrailindex].code, loader.trainLoader, args )
 
 #Memorize best honey source
def memorizeBestSource():
    global best_honey, NectraSource
    for i in range(args.food_number):
        if NectraSource[i].fitness > best_honey.fitness:
            #print(NectraSource[i].fitness, NectraSource[i].code)
            #print(best_honey.fitness, best_honey.code)
            best_honey.code = copy.deepcopy(NectraSource[i].code)
            best_honey.fitness = NectraSource[i].fitness


def main():
    start_epoch = 0
    best_acc = 0.0
    code = []
    
    if args.resume == None:

        test(origin_model, loader.testLoader)

        if args.best_honey == None:

            start_time = time.time()
            
            bee_start_time = time.time()
            
            print('==> Start BeePruning..')

            initilize()

            #memorizeBestSource()

            for cycle in range(args.max_cycle):

                current_time = time.time()
                logger.info(
                    'Search Cycle [{}]\t Best Honey Source {}\tBest Honey Source fitness {:.2f}%\tTime {:.2f}s\n'
                    .format(cycle, best_honey.code, float(best_honey.fitness), (current_time - start_time))
                )
                start_time = time.time()

                sendEmployedBees() 
                  
                calculateProbabilities() 
                  
                sendOnlookerBees()  
                  
                #memorizeBestSource() 
                  
                sendScoutBees() 
                  
                #memorizeBestSource() 

            print('==> BeePruning Complete!')
            bee_end_time = time.time()
            logger.info(
                'Best Honey Source {}\tBest Honey Source fitness {:.2f}%\tTime Used{:.2f}s\n'
                .format(best_honey.code, float(best_honey.fitness), (bee_end_time - bee_start_time))
            )
            #checkpoint.save_honey_model(state)
        else:
            best_honey.code = args.best_honey

        # Model
        print('==> Building model..')
        if args.arch == 'vgg_cifar':
            model = import_module(f'model.{args.arch}').BeeVGG(args.cfg, honeysource=best_honey.code).to(device)
        elif args.arch == 'resnet_cifar':
            model = import_module(f'model.{args.arch}').resnet(args.cfg,honey=best_honey.code).to(device)
        elif args.arch == 'googlenet':
            model = import_module(f'model.{args.arch}').googlenet(honey=best_honey.code).to(device)
        elif args.arch == 'densenet':
            model = import_module(f'model.{args.arch}').densenet(honey=best_honey.code).to(device)

        if args.best_honey_s:
            model.load_state_dict(torch.load(args.best_honey_s))
        else:
            model.load_state_dict(best_honey_state)

        checkpoint.save_honey_model(model.state_dict())

        print(args.random_rule + ' Done!')

        if len(args.gpus) != 1:
            model = nn.DataParallel(model, device_ids=args.gpus)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)
        code = best_honey.code

    else:
        # Model
        resumeckpt = torch.load(args.resume)
        state_dict = resumeckpt['state_dict']
        code = resumeckpt['honey_code']
        print('==> Building model..')
        if args.arch == 'vgg_cifar':
            model = import_module(f'model.{args.arch}').BeeVGG(args.cfg, honeysource=code).to(device)
        elif args.arch == 'resnet_cifar':
            model = import_module(f'model.{args.arch}').resnet(args.cfg,honey=code).to(device)
        elif args.arch == 'googlenet':
            model = import_module(f'model.{args.arch}').googlenet(honey=code).to(device)
        elif args.arch == 'densenet':
            model = import_module(f'model.{args.arch}').densenet(honey=code).to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(resumeckpt['optimizer'])
        scheduler.load_state_dict(resumeckpt['scheduler'])
        start_epoch = resumeckpt['epoch']

        if len(args.gpus) != 1:
            model = nn.DataParallel(model, device_ids=args.gpus)


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
            'epoch': epoch + 1,
            'honey_code': code
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accurary: {:.3f}'.format(float(best_acc)))

if __name__ == '__main__':
    main()
