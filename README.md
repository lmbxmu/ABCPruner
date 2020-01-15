# ABCPruner

Pruning neural network model via Artificial Bee Colony. 

## Requirements

-  Pytorch >= 1.0.1
- CUDA = 10.0.0

## Running Code

### Pre-train Models

Additionally, we provide several pre-trained models used in our experiments.

#### CIFAR-10

| [VGG16](https://drive.google.com/open?id=1pz-_0CCdL-1psIQ545uJ3xT6S_AAnqet) | [ResNet56](https://drive.google.com/open?id=1pt-LgK3kI_4ViXIQWuOP0qmmQa3p2qW5) | [ResNet110](https://drive.google.com/open?id=1Uqg8_J-q2hcsmYTAlRtknCSrkXDqYDMD) |[GoogLeNet](https://drive.google.com/open?id=1YNno621EuTQTVY2cElf8YEue9J4W5BEd) |

#### ImageNet

|[ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) | [ResNet34](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth) |
[ResNet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) | [ResNet152](https://download.pytorch.org/models/resnet152-b121ed2d.pth)|

### BeePruning for Pre-trained model

```shell
python bee_imagenet.py 
--data_path ../data/ImageNet2012 
--honey_model ./pretrain/resnet18.pth 
--job_dir ./experiment/resnet_imagenet 
--arch resnet
--cfg resnet18
--lr 0.01 
--lr_decay_step 75 112 
--num_epochs 150 
--gpus 0 
--calfitness_epoch 2 
--max_cycle 50 
--max_preserve 9 
--food_number 10 
--food_limit 5 
--random_rule random_pretrain
```



### BeePruning for Pre-trained model（with LabelSmooth）

```shell
python bee_imagenet_smooth.py 
--data_path ../data/ImageNet2012 
--honey_model ./pretrain/resnet18.pth 
--job_dir ./experiment/resnet_imagenet 
--arch resnet
--cfg resnet18
--lr 0.01 
--lr_decay_step 75 112 
--num_epochs 150 
--gpus 0 
--calfitness_epoch 2 
--max_cycle 50 
--max_preserve 9 
--food_number 10 
--food_limit 5 
--random_rule random_pretrain
```

### Get FLOPS

```shell
python get_flops.py 
--data_set cifar10 
--arch resnet_cifar 
--cfg resnet56
--honey 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
```

## Other Arguments

```shell
optional arguments:
  -h, --help            show this help message and exit
  --gpus GPUS [GPUS ...]
                        Select gpu_id to use. default:[0]
  --data_set DATA_SET   Select dataset to train. default:cifar10
  --data_path DATA_PATH
                        The dictionary where the input is stored.
                        default:/home/lishaojie/data/cifar10/
  --job_dir JOB_DIR     The directory where the summaries will be stored.
                        default:./experiments
  --reset               Reset the directory?
  --resume RESUME       Load the model from the specified checkpoint.
  --refine REFINE       Path to the model to be fine tuned.
  --arch ARCH           Architecture of model. default:vgg,resnet,googlenet,densenet
  --cfg CFG             Detail architecuture of model. default:vgg16, resnet18/34/50(imagenet),resnet56/110(cifar),googlenet,densenet
  --num_epochs NUM_EPOCHS
                        The num of epochs to train. default:150
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size for training. default:128
  --eval_batch_size EVAL_BATCH_SIZE
                        Batch size for validation. default:100
  --momentum MOMENTUM   Momentum for MomentumOptimizer. default:0.9
  --lr LR               Learning rate for train. default:1e-2
  --lr_decay_step LR_DECAY_STEP [LR_DECAY_STEP ...]
                        the iterval of learn rate. default:50, 100
  --weight_decay WEIGHT_DECAY
                        The weight decay of loss. default:5e-4
  --random_rule RANDOM_RULE
                        Weight initialization criterion after random clipping.
                        default:default
                        optional:default,random_pretrain,l1_pretrain
  --test_only           Test only?
  --honey_model         Path to the model wait for Beepruning. default:None
  --calfitness_epoch    Calculate fitness of honey source: training epochs. default:2
  --max_cycle           Search for best pruning plan times. default:10
  --food_number         number of food to search. default:10
  --food_limit          Beyond this limit, the bee has not been renewed to become a scout bee default:5
  --honeychange_num     Number of codes that the nectar source changes each time default:2
  --best_honey          If this hyper-parameter exists, skip bee-pruning and fine-tune from this prune method default:None
  --best_honey_s        Path to the best_honey default:None
  --best_honey_past     If you want to load a resume without honey code, input your honey hode into this hyper-parameter default:None

  --honey               get flops and params of a model with specified honey(prune plan )
  --from_scratch        if this parameter exist, train from scratch 
  --warm_up             if this parameter exist, use warm up lr like DALI
  

```
