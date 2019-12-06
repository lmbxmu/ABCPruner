# BeePruning

Pruning neural network model via BeePruning. 

## Requirements

-  Pytorch >= 1.0.1
- CUDA = 10.0.0

## Running Code

### Pre-train Models

Additionally, we provide several  pre-trained models used in our experiments.

#### CIFAR-10

| [VGG16](https://drive.google.com/open?id=1iqcLZyMTnciVLiKOHNaKbeXixK0KOzuX) | [ResNet18](https://drive.google.com/open?id=1NuzORsV2O8QfkbCV72EtKp1iZQCBAnB1) | [ResNet34](https://drive.google.com/open?id=1_P3zNbtTpery4jjAu4o43-WjlPc6ot81) | [ResNet50](https://drive.google.com/open?id=1MihR1PxF7ibhOPnnC-FAQci9KC2tVoKL) | [ResNet101](https://drive.google.com/open?id=14q3u3fYeFUHcBRKUtx_MGIjBtJxfTVRM) | [ResNet152](https://drive.google.com/open?id=1FQKqA2WrD0o0qxlhPRC1RXImppoBC91b) | 

#### ImageNet

| [VGG16](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth) | 
|[ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) | [ResNet34](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth) | [ResNet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) | [ResNet152](https://download.pytorch.org/models/resnet152-b121ed2d.pth)|
|[GoogLeNet](https://download.pytorch.org/models/googlenet-1378be20.pth)|
|[DenseNet121](https://drive.google.com/open?id=1-ZZu8yGmh518F6621BvHwBZ7NV17wf-9)|[DenseNet161](https://drive.google.com/open?id=1lNWiyyeQKtsldO7iFNmQ11WLNUNH22Jr)|[DenseNet169](https://drive.google.com/open?id=10iScGCR4QY6ZkghATkEaa61-F8buW3fB)|[DenseNet201](https://drive.google.com/open?id=1DZytePACQJyXbgLX_KIUDJRHAerUo4OT)|

### Train from scratch

```shell
python main.py 
--arch vgg 
--cfg vgg16 
--data_set cifar10 
--data_path /data/cifar10
--gpus 0 
--job_dir ./experiment/vgg16
```

### BeePruning for Pre-trained model

```shell
python main_bee_VGG.py 
--data_set cifar10 
--data_path ../../data/cifar10 
--honey_model ./experiment/vgg16/baseline/checkpoint/model_best.pt
--job_dir ./experiment/vgg16/finetune25  
--arch vgg 
--cfg vgg16 
--lr 0.01
--lr_decay_step 75 112
--num_epochs 150 
--gpus 0
--calfitness_epoch 10
--maxcycle 1000
--max_preserve 9
--food_number 50
--food_dimension 13
--food_limit 5

```



### Other optional arguments

```
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
  --arch ARCH           Architecture of model. default:vgg
  --cfg CFG             Detail architecuture of model. default:vgg16
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
```

