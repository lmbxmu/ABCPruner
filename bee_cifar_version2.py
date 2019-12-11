import torch
import torch.nn as nn
import torch.optim as optim
#from model.googlenet import Inception
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


device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
checkpoint = utils.checkpoint(args)
loss_func = nn.CrossEntropyLoss()

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
elif args.arch == 'resnet':
    pass
elif args.arch == 'googlenet':
    pass
elif args.arch == 'densenet':
    pass

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
        self.selected = []
        self.rfitness = 0 
        self.trail = 0

#Initilize global element
best_honey = BeeGroup()
NectraSource = []
EmployedBee = []
OnLooker = []

#load pre-train params
def load_vgg_honey_model(model, random_rule):
    #print(ckpt['state_dict'])
    global oristate_dict
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer
    selected_index = []

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

                selected_index.append(copy.deepcopy(select_index))

                last_select_index = select_index

            else:
                state_dict[name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)
    return selected_index

#load pre-train params
def load_best_honey_model(model, random_rule):
    #print(ckpt['state_dict'])
    global oristate_dict
    global best_honey
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer
    selected_index = []
    index = 0

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']
            
            if last_select_index is not None:
                for index_i, i in enumerate(selected_index[index]):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
            else:
                for index_i, i in enumerate(selected_index[index]):
                    state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

            last_select_index = select_index
            index += 1


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

    if args.arch == 'vgg_cifar':
        model = import_module(f'model.{args.arch}').BeeVGG(args.cfg,honeysource=honey).to(device)
        selected_index = load_vgg_honey_model(model, args.random_rule)
    elif args.arch == 'resnet':
        pass
    elif args.arch == 'googlenet':
        pass
    elif args.arch == 'densenet':
        pass

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

    return fit_accurary.avg, selected_index



#Initilize Bee-Pruning
def initilize():
    print('==> Initilizing Honey_model..')
    global best_honey, NectraSource, EmployedBee, OnLooker

    for i in range(args.food_number):
        NectraSource.append(copy.deepcopy(BeeGroup()))
        EmployedBee.append(copy.deepcopy(BeeGroup()))
        OnLooker.append(copy.deepcopy(BeeGroup()))
        for j in range(args.food_dimension):
            NectraSource[i].code.append(copy.deepcopy(random.randint(1,args.max_preserve)))

        #initilize honey souce

        NectraSource[i].fitness, NectraSource[i].selected = calculationFitness(NectraSource[i].code, loader.trainLoader, args)
        NectraSource[i].rfitness = 0
        NectraSource[i].trail = 0

        #initilize employed bee  
        EmployedBee[i].code = copy.deepcopy(NectraSource[i].code)
        EmployedBee[i].fitness=NectraSource[i].fitness 
        EmployedBee[i].selected = NectraSource[i].selected
        EmployedBee[i].rfitness=NectraSource[i].rfitness 
        EmployedBee[i].trail=NectraSource[i].trail

        #initilize onlooker 
        OnLooker[i].code = copy.deepcopy(NectraSource[i].code)
        OnLooker[i].fitness=NectraSource[i].fitness 
        OnLooker[i].selected = NectraSource[i].selected
        OnLooker[i].rfitness=NectraSource[i].rfitness 
        OnLooker[i].trail=NectraSource[i].trail

    #initilize best honey
    best_honey.code = copy.deepcopy(NectraSource[0].code)
    best_honey.fitness = NectraSource[0].fitness
    best_honey[i].selected = NectraSource[i].selected
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

        param2change = np.random.randint(0, args.food_dimension-1, args.honeychange_num)
        R = np.random.uniform(-1, 1, args.honeychange_num)
        for j in range(args.honeychange_num):
            EmployedBee[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]]+ R[j]*(NectraSource[i].code[param2change[j]]-NectraSource[k].code[param2change[j]]))
            if EmployedBee[i].code[param2change[j]] < 1:
                EmployedBee[i].code[param2change[j]] = 1
            if EmployedBee[i].code[param2change[j]] > args.max_preserve:
                EmployedBee[i].code[param2change[j]] = args.max_preserve

        EmployedBee[i].fitness, EmployedBee[i].selected = calculationFitness(EmployedBee[i].code, loader.trainLoader, args)

        if EmployedBee[i].fitness > NectraSource[i].fitness:                
            NectraSource[i].code = copy.deepcopy(EmployedBee[i].code)
            NectraSource[i].selected = EmployedBee[i].selected             
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

            param2change = np.random.randint(0, args.food_dimension-1, args.honeychange_num)
            R = np.random.uniform(-1, 1, args.honeychange_num)
            for j in range(args.honeychange_num):
                OnLooker[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]]+ R[j]*(NectraSource[i].code[param2change[j]]-NectraSource[k].code[param2change[j]]))
                if OnLooker[i].code[param2change[j]] < 1:
                    OnLooker[i].code[param2change[j]] = 1
                if OnLooker[i].code[param2change[j]] > args.max_preserve:
                    OnLooker[i].code[param2change[j]] = args.max_preserve

            OnLooker[i].fitness, OnLooker[maxtrailindex].selected= calculationFitness(OnLooker[i].code, loader.trainLoader, args)

            if OnLooker[i].fitness > NectraSource[i].fitness:                
                NectraSource[i].code = copy.deepcopy(OnLooker[i].code)              
                NectraSource[i].trail = 0  
                NectraSource[i].selected = Onlooker[i].selected
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
        for j in range(args.food_dimension):
            R = random.uniform(0,1)
            NectraSource[maxtrailindex].code[j] = int(R * args.max_preserve)
            if NectraSource[maxtrailindex].code[j] == 0:
                NectraSource[maxtrailindex].code[j] += 1
        NectraSource[maxtrailindex].trail = 0
        NectraSource[maxtrailindex].fitness, NectraSource[maxtrailindex].selected= calculationFitness(NectraSource[maxtrailindex].code, loader.trainLoader, args )
 
 #Memorize best honey source
def memorizeBestSource():
    global best_honey, NectraSource
    for i in range(args.food_number):
        if NectraSource[i].fitness > best_honey.fitness:
            #print(NectraSource[i].fitness, NectraSource[i].code)
            #print(best_honey.fitness, best_honey.code)
            best_honey.code = copy.deepcopy(NectraSource[i].code)
            best_honey.selected = NectraSource[i].selected
            best_honey.fitness = NectraSource[i].fitness


def main():
    start_epoch = 0
    best_acc = 0.0

    test(origin_model, loader.testLoader)

    if args.best_honey == None:

        start_time = time.time()
        
        bee_start_time = time.time()
        
        print('==> Start BeePruning..')

        initilize()

        memorizeBestSource()

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
              
            memorizeBestSource() 
              
            sendScoutBees() 
              
            memorizeBestSource() 

        print('==> BeePruning Complete!')
        bee_end_time = time.time()
        logger.info(
            'Best Honey Source {}\tBest Honey Source fitness {:.2f}%\tTime Used{:.2f}s\n'
            .format(best_honey.code, float(best_honey.fitness), (bee_end_time - bee_start_time))
        )
        checkpoint.save_honey_model(state)
    else:
        best_honey.code = args.best_honey

    # Model
    print('==> Building model..')
    if args.arch == 'vgg':
        model = import_module(f'model.{args.arch}').BeeVGG(args.cfg, honeysource=best_honey.code).to(device)
        load_best_honey_model(model)
    elif args.arch == 'resnet':
        pass
    elif args.arch == 'googlenet':
        pass
    elif args.arch == 'densenet':
        pass

    print(args.random_rule + ' Done!')

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

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
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accurary: {:.3f}'.format(float(best_acc)))

if __name__ == '__main__':
    main()
