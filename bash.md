python bee_imagenet.py  --data_path ../ImageNet2012 --job_dir ./experiment/from_scratch_1 --arch resnet --cfg resnet18 --lr 0.01 --lr_decay_step 60 90 --num_epochs 150 --gpus 0 3 --from_scratch --train_batch_size 256 --eval_batch_size 256 


python bee_imagenet.py  --data_path /home/sda1/data/ImageNet2012 --job_dir ./experiment/from_scratch_2 --arch resnet --cfg resnet18 --lr 0.01 --lr_decay_step 10 60 90 --num_epochs 120  --gpus 0 1 2 --from_scratch --train_batch_size 256 --eval_batch_size 256 


python imagenet_withoutdali.py  --data_path ../ImageNet2012 --job_dir ./experiment/from_scratch_withoutdali --arch resnet --cfg resnet18 --lr 0.01 --lr_decay_step 10 60 90 --num_epochs 120  --gpus 0 1 2 3 --from_scratch --train_batch_size 256 --eval_batch_size 256


python bee_imagenet.py --data_path /media/disk2/zyc/ImageNet2012 --honey_model ./experience/resnet/baseline/checkpoint/resnet18.pth --job_dir ./experiment/resnet_imagenet_test_1 --arch resnet --cfg resnet18 --lr 0.001 --lr_decay_step 10 20 --weight_decay 0.0005 --num_epochs 30 --gpus 0 1 2 --calfitness_epoch 1 --max_cycle 1 --max_preserve 9 --food_number 2 --food_limit 5 --random_rule random_pretrain --train_batch_size 512 --eval_batch_size 512

python bee_imagenet.py --data_path /home/sda1/data/ImageNet2012 --honey_model ./experience/resnet/baseline/checkpoint/resnet18.pth --job_dir ./experiment/resnet_imagenet_test_2 --arch resnet --cfg resnet18 --lr 0.001 --lr_decay_step 10 20 --weight_decay 0.0005 --num_epochs 30 --gpus 0 1 2 --calfitness_epoch 1 --max_cycle 1 --max_preserve 9 --food_number 2 --food_limit 5 --random_rule random_pretrain --train_batch_size 256 --eval_batch_size 256