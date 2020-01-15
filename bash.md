python bee_imagenet.py  --data_path ../ImageNet2012 --job_dir ./experiment/from_scratch_7 --arch resnet --cfg resnet18 --lr 0.1 --lr_decay_step 60 90 --num_epochs 150 --gpus 2 3 --from_scratch --train_batch_size 256 --eval_batch_size 256 


python bee_imagenet.py  --data_path /home/sda1/data/ImageNet2012 --job_dir ./experiment/from_scratch_2 --arch resnet --cfg resnet18 --lr 0.01 --lr_decay_step 10 60 90 --num_epochs 120  --gpus 0 1 2 --from_scratch --train_batch_size 256 --eval_batch_size 256 


python imagenet_withoutdali.py  --data_path ../ImageNet2012 --job_dir ./experiment/from_scratch_withoutdali --arch resnet --cfg resnet18 --lr 0.01 --lr_decay_step 10 60 90 --num_epochs 120  --gpus 0 1 2 3 --from_scratch --train_batch_size 256 --eval_batch_size 256


python bee_imagenet.py --data_path /media/disk2/zyc/ImageNet2012 --honey_model ./experience/resnet/baseline/checkpoint/resnet18.pth --job_dir ./experiment/resnet_imagenet_test_1 --arch resnet --cfg resnet18 --lr 0.001 --lr_decay_step 10 20 --weight_decay 0.0005 --num_epochs 30 --gpus 0 1 2 --calfitness_epoch 1 --max_cycle 1 --max_preserve 9 --food_number 2 --food_limit 5 --random_rule random_pretrain --train_batch_size 512 --eval_batch_size 512

python bee_imagenet.py --data_path /home/sda1/data/ImageNet2012 --honey_model ./experience/resnet/baseline/checkpoint/resnet18.pth --job_dir ./experiment/resnet_imagenet_test_2 --arch resnet --cfg resnet18 --lr 0.001 --lr_decay_step 10 20 --weight_decay 0.0005 --num_epochs 30 --gpus 0 1 2 --calfitness_epoch 1 --max_cycle 1 --max_preserve 9 --food_number 2 --food_limit 5 --random_rule random_pretrain --train_batch_size 256 --eval_batch_size 256

python DALI.py --gpus 0 1 2 -a resnet18 /media/disk2/zyc/ImageNet2012/ILSVRC2012_img_train /media/disk2/zyc/ImageNet2012/val


python bee_imagenet_dali_reset.py --data_path /media/disk2/zyc/ImageNet2012 --honey_model ./experience/resnet/baseline/checkpoint/resnet18.pth --job_dir ./experiment/resnet_imagenetdali --arch resnet --cfg resnet18 --lr 0.1 --weight_decay 0.0001 --num_epochs 90 --gpus 0 1 --calfitness_epoch 1 --max_cycle 1 --max_preserve 9 --food_number 2 --food_limit 5 --random_rule random_pretrain --train_batch_size 256 --eval_batch_size 256



python bee_imagenet_dali_reset.py --data_path ../ImageNet2012 --honey_model ./experience/resnet/baseline/checkpoint/resnet18.pth --job_dir ./experiment/resnet_imagenetdali_reser --arch resnet --cfg resnet18 --lr 0.1 --weight_decay 0.0001 --num_epochs 90 --gpus 0 1 2 3 --calfitness_epoch 1 --max_cycle 1 --max_preserve 9 --food_number 2 --food_limit 5 --random_rule random_pretrain --train_batch_size 256 --eval_batch_size 256


python bee_imagenet_dali_reset.py --data_path /media/disk2/zyc/ImageNet2012 --honey_model ./experience/resnet/baseline/checkpoint/resnet18.pth --job_dir ./experiment/resnet_imagenetdali_reser --arch resnet --cfg resnet18 --lr 0.1 --weight_decay 0.0001 --num_epochs 90 --gpus 0 1  --calfitness_epoch 1 --max_cycle 1 --max_preserve 9 --food_number 2 --food_limit 5 --random_rule random_pretrain --train_batch_size 256 --eval_batch_size 256


python bee_imagenet_smooth.py --data_path /media/disk2/zyc/ImageNet2012 --honey_model ./experience/resnet/baseline/checkpoint/resnet18.pth --job_dir ./experiment/resnet_imagenetdali_reser --arch resnet --cfg resnet18 --lr 0.1 --weight_decay 0.0001 --num_epochs 32 --lr_decay_step 8 16 24 --gpus 0 1  --calfitness_epoch 1 --max_cycle 1 --max_preserve 9 --food_number 2 --food_limit 5 --random_rule random_pretrain --train_batch_size 256 --eval_batch_size 256


22号服务器在跑bee lr从epoch2开始，测试会不会出现前几个epoch掉精度情况

19号服务器在测reset_train_from scratch

79号服务器在测bee lr从epoch0开始，测试会不会出现前几个epoch掉精度情况，之后会恢复正常训练吗？



conv_num_cfg = {
    'resnet18' : 8,
    'resnet34' : 16,
    'resnet50' : 16,
    'resnet101' : 33,
    'resnet152' : 50 
}


python get_flops.py --data_set imagenet --arch resnet --cfg resnet18 --honey 4 4 4 4 4 4 4 4 

python get_flops.py --data_set imagenet --arch resnet --cfg resnet18 --honey 5 5 5 5 5 5 5 5

python get_flops.py --data_set imagenet --arch resnet --cfg resnet18 --honey 6 6 6 6 6 6 6 6

python get_flops.py --data_set imagenet --arch resnet --cfg resnet18 --honey 7 7 7 7 7 7 7 7

python get_flops.py --data_set imagenet --arch resnet --cfg resnet18 --honey 8 8 8 8 8 8 8 8

python get_flops.py --data_set imagenet --arch resnet --cfg resnet18 --honey 9 9 9 9 9 9 9 9


python get_flops.py --data_set imagenet --arch resnet --cfg resnet34 --honey 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 

python get_flops.py --data_set imagenet --arch resnet --cfg resnet34 --honey 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5

python get_flops.py --data_set imagenet --arch resnet --cfg resnet34 --honey 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6

python get_flops.py --data_set imagenet --arch resnet --cfg resnet34 --honey 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7

python get_flops.py --data_set imagenet --arch resnet --cfg resnet34 --honey 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8

python get_flops.py --data_set imagenet --arch resnet --cfg resnet34 --honey 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9


python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9



python get_flops.py --data_set imagenet --arch resnet --cfg resnet101 --honey 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4

python get_flops.py --data_set imagenet --arch resnet --cfg resnet101 --honey 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5

python get_flops.py --data_set imagenet --arch resnet --cfg resnet101 --honey 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6

python get_flops.py --data_set imagenet --arch resnet --cfg resnet101 --honey 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7

python get_flops.py --data_set imagenet --arch resnet --cfg resnet101 --honey 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8

python get_flops.py --data_set imagenet --arch resnet --cfg resnet101 --honey 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9



python get_flops.py --data_set imagenet --arch resnet --cfg resnet152 --honey 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4

python get_flops.py --data_set imagenet --arch resnet --cfg resnet152 --honey 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5

python get_flops.py --data_set imagenet --arch resnet --cfg resnet152 --honey 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6

python get_flops.py --data_set imagenet --arch resnet --cfg resnet152 --honey 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7

python get_flops.py --data_set imagenet --arch resnet --cfg resnet152 --honey 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8

python get_flops.py --data_set imagenet --arch resnet --cfg resnet152 --honey 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9


统计res56在不同的max_preserve对应下的accuracy,   channel/flops/params的pruning rate.做成一个统计图

max_preserve                   bestsource                                   acc         channel 2032    flop 127.62    param 0.85
1           1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1           87.61       1114 45.18%     12.02 90.58    0.08 90.45
2           1 1 1 1 1 2 1 1 1 1 1 1 2 2 2 1 2 1 1 1 2 2 2 2 2 2 1           88.84		1164 42.72%     17.06 86.63	   0.13 84.69
3           2 3 2 2 1 2 3 1 2 3 2 1 1 3 2 1 1 1 2 1 1 1 3 3 3 2 2           90.38		1205 40.70		23.36 81.69	   0.16 81.27
4           4 2 3 3 4 1 3 1 4 2 3 4 3 2 2 4 4 3 3 2 3 4 3 3 1 2 2           91.57		1283 36.86%     34.05 73.32	   0.22 74.35
5           5 3 2 1 2 4 1 3 4 5 5 5 1 3 1 5 1 2 5 1 2 1 3 3 5 2 2           92.04       1298 36.12		35.29 72.34	   0.23	73.50
6           3 3 2 6 5 1 4 2 4 2 4 3 3 6 5 3 6 2 4 1 6 3 6 5 3 6 2           92.60		1400 31.10		46.40 63.64	   0.33 61.26
7           5 7 3 6 6 6 7 6 3 6 2 7 1 2 1 5 5 5 7 4 7 6 2 4 5 3 6           93.23		1482 27.07		58.54 54.13	   0.39 54.12
8           8 5 6 6 6 6 8 4 3 6 8 7 6 7 1 3 2 3 8 6 2 8 8 5 5 8 1           93.39		1560 23.23      67.09 47.73	   0.46 46.11
9           7 4 6 2 1 3 6 7 1 8 4 6 3 6 5 6 4 4 6 4 4 5 5 1 8 7 1           93.1		1481 27.12 		56.76 55.52	   0.39 54.74
10          2 2 7 5 9 1 8 5 6 9 3 8 10 4 2 8 8 7 3 10 6 5 8 9 10 8 9        93.52		1710 15.85		80.15 37.20	   0.62 27.360,	


python get_flops.py --data_set cifar10 --arch resnet_cifar --cfg resnet56 --honey 1 1 1 1 1 2 1 1 1 1 1 1 2 2 2 1 2 1 1 1 2 2 2 2 2 2 1

python get_flops.py --data_set cifar10 --arch resnet_cifar --cfg resnet56 --honey 2 3 2 2 1 2 3 1 2 3 2 1 1 3 2 1 1 1 2 1 1 1 3 3 3 2 2

python get_flops.py --data_set cifar10 --arch resnet_cifar --cfg resnet56 --honey 4 2 3 3 4 1 3 1 4 2 3 4 3 2 2 4 4 3 3 2 3 4 3 3 1 2 2

python get_flops.py --data_set cifar10 --arch resnet_cifar --cfg resnet56 --honey 5 3 2 1 2 4 1 3 4 5 5 5 1 3 1 5 1 2 5 1 2 1 3 3 5 2 2

python get_flops.py --data_set cifar10 --arch resnet_cifar --cfg resnet56 --honey 3 3 2 6 5 1 4 2 4 2 4 3 3 6 5 3 6 2 4 1 6 3 6 5 3 6 2

python get_flops.py --data_set cifar10 --arch resnet_cifar --cfg resnet56 --honey 5 7 3 6 6 6 7 6 3 6 2 7 1 2 1 5 5 5 7 4 7 6 2 4 5 3 6

python get_flops.py --data_set cifar10 --arch resnet_cifar --cfg resnet56 --honey 8 5 6 6 6 6 8 4 3 6 8 7 6 7 1 3 2 3 8 6 2 8 8 5 5 8 1

python get_flops.py --data_set cifar10 --arch resnet_cifar --cfg resnet56 --honey 7 4 6 2 1 3 6 7 1 8 4 6 3 6 5 6 4 4 6 4 4 5 5 1 8 7 1

python get_flops.py --data_set cifar10 --arch resnet_cifar --cfg resnet56 --honey 2 2 7 5 9 1 8 5 6 9 3 8 10 4 2 8 8 7 3 10 6 5 8 9 10 8 9






python get_flops.py --data_set imagenet --arch resnet --cfg resnet18 --honey '1, 3, 3, 4, 4, 5, 5, 4'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet18 --honey '1, 6, 1, 4, 4, 1, 6, 5'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet18 --honey '7, 2, 4, 7, 4, 6, 6, 5'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet18 --honey '8, 2, 5, 2, 6, 1, 5, 8'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet18 --honey '4, 7, 6, 4, 3, 3, 3, 9'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet18 --honey '3, 1, 4, 3, 8, 4, 9, 9'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet34 --honey '1, 4, 4, 1, 3, 3, 5, 5, 5, 5, 3, 4, 4, 5, 5, 5'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet34 --honey '1, 3, 2, 2, 4, 1, 6, 3, 4, 3, 6, 6, 1, 4, 4, 6'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet34 --honey '5, 6, 2, 3, 3, 3, 1, 6, 1, 6, 5, 2, 1, 6, 4, 5'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet34 --honey '7, 6, 5, 5, 5, 1, 5, 6, 7, 6, 6, 5, 2, 5, 6, 2'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet34 --honey '3, 5, 6, 5, 5, 7, 1, 6, 2, 2, 7, 5, 5, 5, 5, 4'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet34 --honey '3, 1, 2, 4, 2, 3, 1, 1, 2, 1, 3, 2, 3, 5, 5, 5'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet34 --honey '8, 7, 2, 3, 6, 4, 5, 5, 1, 2, 4, 3, 3, 6, 8, 8'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet34 --honey '6, 5, 8, 2, 6, 7, 8, 7, 2, 9, 8, 9, 4, 4, 3, 3'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey '2, 2, 2, 1, 1, 2, 1, 3, 2, 3, 2, 2, 3, 3, 1, 2'	

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey '2, 4, 2, 1, 1, 3, 3, 3, 3, 1, 3, 4, 1, 3, 4, 3'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey '3, 4, 3, 5, 1, 3, 1, 3, 2, 4, 3, 5, 5, 3, 2, 4'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey '3, 2, 5, 2, 4, 2, 1, 4, 4, 1, 4, 1, 5, 3, 3, 4'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey '3, 6, 1, 1, 4, 5, 1, 4, 4, 6, 5, 3, 4, 2, 1, 5'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey '3, 2, 3, 1, 3, 1, 3, 6, 6, 1, 5, 3, 6, 5, 2, 6'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey '7, 2, 7, 4, 3, 7, 7, 2, 6, 5, 5, 2, 7, 3, 6, 3'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey '2, 1, 6, 5, 2, 3, 4, 7, 4, 2, 4, 5, 4, 7, 1, 1'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey '1, 2, 8, 8, 2, 5, 7, 5, 5, 5, 1, 8, 7, 2, 5, 5'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey '3, 4, 6, 4, 6, 1, 5, 9, 6, 8, 9, 4, 3, 6, 4, 2'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet50 --honey '3, 2, 7, 3, 10, 7, 9, 10, 9, 2, 5, 8, 3, 10, 10, 2'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet101 --honey '3, 5, 2, 3, 2, 2, 5, 4, 1, 3, 5, 4, 3, 1, 3, 5, 5, 3, 1, 5, 1, 5, 4, 4, 2, 3, 1, 4, 1, 2, 5, 1, 1'


python get_flops.py --data_set imagenet --arch resnet --cfg resnet101 --honey '5, 5, 4, 4, 4, 2, 3, 1, 6, 7, 4, 1, 1, 4, 6, 5, 6, 4, 5, 6, 3, 7, 1, 6, 1, 5, 7, 3, 3, 7, 6, 7, 1'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet101 --honey '6, 6, 3, 6, 5, 5, 3, 3, 2, 7, 7, 1, 5, 1, 7, 1, 7, 3, 4, 4, 3, 7, 7, 1, 7, 3, 5, 6, 6, 3, 2, 4, 3'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet101 --honey '5, 5, 1, 6, 3, 4, 3, 3, 6, 2, 5, 2, 1, 6, 3, 1, 3, 1, 1, 2, 3, 4, 1, 6, 2, 1, 2, 4, 4, 4, 1, 4, 4'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet101 --honey '4, 2, 3, 7, 7, 8, 4, 5, 6, 4, 6, 7, 4, 6, 1, 3, 5, 4, 5, 1, 1, 8, 5, 2, 3, 6, 8, 6, 2, 8, 8, 1, 1'


python get_flops.py --data_set imagenet --arch resnet --cfg resnet152 --honey '1, 3, 3, 3, 1, 1, 3, 2, 3, 3, 5, 4, 5, 1, 5, 3, 3, 5, 4, 1, 3, 5, 3, 5, 5, 3, 3, 2, 5, 3, 4, 3, 4, 4, 5, 1, 5, 4, 1, 5, 5, 3, 3, 1, 3, 5, 1, 5, 3, 5'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet152 --honey '1, 2, 1, 3, 1, 5, 2, 2, 4, 6, 1, 5, 6, 6, 5, 4, 3, 3, 2, 4, 6, 4, 4, 2, 6, 2, 2, 1, 2, 4, 4, 3, 6, 3, 1, 4, 3, 5, 4, 6, 6, 1, 6, 5, 2, 6, 5, 2, 1, 6'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet152 --honey '7, 6, 6, 6, 1, 6, 7, 4, 7, 2, 3, 1, 1, 4, 1, 4, 6, 5, 7, 1, 7, 2, 3, 7, 6, 6, 7, 7, 2, 3, 2, 1, 6, 1, 5, 2, 7, 7, 4, 2, 6, 1, 3, 7, 6, 4, 7, 7, 6, 4'

python get_flops.py --data_set imagenet --arch resnet --cfg resnet152 --honey '5, 2, 1, 3, 7, 5, 2, 3, 4, 6, 5, 4, 6, 6, 1, 5, 6, 3, 2, 6, 1, 1, 6, 5, 1, 5, 5, 5, 3, 5, 4, 7, 6, 1, 5, 5, 1, 3, 1, 6, 3, 6, 4, 7, 2, 3, 6, 1, 4, 5'









