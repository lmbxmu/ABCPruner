python bee_imagenet.py  --data_path ../ImageNet2012 --job_dir ./experiment/from_scratch_1 --arch resnet --cfg resnet18 --lr 0.01 --lr_decay_step 60 90 --num_epochs 150 --gpus 0 3 --from_scratch --train_batch_size 256 --eval_batch_size 256 


python bee_imagenet.py  --data_path /home/sda1/data/ImageNet2012 --job_dir ./experiment/from_scratch_2 --arch resnet --cfg resnet18 --lr 0.01 --lr_decay_step 10 60 90 --num_epochs 120  --gpus 0 1 2 --from_scratch --train_batch_size 256 --eval_batch_size 256 