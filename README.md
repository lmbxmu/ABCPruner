# BeePruning

Pruning neural network model via BeePruning. 

## Requirements

-  Pytorch >= 1.0.1
- CUDA = 10.0.0

## Running Code

### Pre-train Models

Additionally, we provide several pre-trained models used in our experiments.

#### CIFAR-10

| [VGG16](https://drive.google.com/open?id=1pz-_0CCdL-1psIQ545uJ3xT6S_AAnqet) | [ResNet56](https://drive.google.com/open?id=1pt-LgK3kI_4ViXIQWuOP0qmmQa3p2qW5) | [ResNet110](https://drive.google.com/open?id=1Uqg8_J-q2hcsmYTAlRtknCSrkXDqYDMD) |[GoogLeNet](https://drive.google.com/open?id=1YNno621EuTQTVY2cElf8YEue9J4W5BEd) | [DenseNet40](https://drive.google.com/open?id=1TV_b98le-R0sDIkhc5pfrO6zn4uggOWl) | 

#### ImageNet

|[ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) | [ResNet34](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth) |

### Train from scratch

```shell
python main.py 
--arch vgg 
--cfg vgg16 
--data_set cifar10 
--data_path /home/lmb/cvpr_vgg2/data
--gpus 0 
--job_dir ./experiment/vgg16
```

### BeePruning for Pre-trained model ----Cifar resnet

```shell

python bee_cifar.py
--data_set cifar10 
--data_path /home/lmb/cvpr_vgg2/data 
--honey_model ./experience/resnet/baseline/checkpoint/resnet_56.pt 
--job_dir ./experiment/resnet56 
--arch resnet_cifar 
--cfg resnet56 
--lr 0.01 
--lr_decay_step 75 112 
--num_epochs 150 
--gpus 0 
--calfitness_epoch 2 
--max_cycle 50 
--max_preserve 9 
--food_number 10 
--food_dimension 13 
--food_limit 5 
--random_rule random_pretrain

```

### BeePruning for Pre-trained model ----Resnet Imagenet

```shell
python bee_imagenet.py 
--data_path /home/sda4/data/ImageNet2012 
--honey_model ./experience/resnet/baseline/checkpoint/resnet18.pth 
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
--food_dimension 13 
--food_limit 5 
--random_rule random_pretrain


```



### BeePruning for Pre-trained model ----Cifar VGG

```shell
python bee_cifar.py 
--data_set cifar10 
--data_path /home/lmb/cvpr_vgg2/data 
--honey_model ./experience/vgg16/baseline/checkpoint/vgg16_cifar10.pt  
--job_dir ./experiment/vgg16 
--arch vgg_cifar 
--cfg vgg16 
--lr 0.01 
--lr_decay_step 75 112 
--num_epochs 150 
--gpus 0 
--calfitness_epoch 2 
--max_cycle 50 
--max_preserve 9 
--food_number 10 
--food_dimension 13 
--food_limit 5 
--random_rule random_pretrain



```
### BeePruning for Pre-trained model ----VGG Imagenet

```shell
python bee_imagenet.py 
--data_path /home/sda4/data/ImageNet2012 
--honey_model ./experience/vgg16/baseline/checkpoint/vgg16_imagenet.pth 
--job_dir ./experiment/vgg16_imagenet 
--arch vgg 
--cfg vgg16 
--lr 0.01 
--lr_decay_step 75 112 
--num_epochs 150 
--gpus 0 
--calfitness_epoch 2 
--max_cycle 50 
--max_preserve 9 
--food_number 10 
--food_dimension 13 
--food_limit 5 
--random_rule random_pretrain


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
  --start_conv START_CONV
                        The index of Conv to start sketch, index starts from
                        0. default:1
  --sketch_rate SKETCH_RATE
                        The rate of each sketch conv. default:None
  --sketch_model SKETCH_MODEL
                        Path to the model wait for sketch. default:None
  --sketch_bn SKETCH_BN
                        Whether the BN weights are sketched or not?
                        default:False
  --weight_norm_method WEIGHT_NORM_METHOD
                        Select the weight norm method. default:None
                        Optional:max,sum,l2,l1,l2_2,2max
  --filter_norm FILTER_NORM
                        Filter level normalization or not? default:False
  --sketch_lastconv SKETCH_LASTCONV
                        Is the last layer of convolution sketched?
                        default:True
  --random_rule RANDOM_RULE
                        Weight initialization criterion after random clipping.
                        default:default
                        optional:default,random_pretrain,l1_pretrain
  --test_only           Test only?

```
