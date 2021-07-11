# Meta Pseudo Labels
This is an unofficial PyTorch implementation of [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580) specified on our task .
The original unofficial PyTorch implementation is [here](https://github.com/kekmodel/MPL-pytorch)
The official Tensorflow implementation is [here](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels).


## Usage

Train the model on HBS data set

```
python main.py --seed 42 --name HBS3.5k --expand-labels --dataset hbs --num-classes 9 --total-steps 100000 --eval-step 1000 --randaug 2 16 --batch-size 18 --teacher_lr 0.025 --student_lr 0.025 --weight-decay 5e-4 --ema 0.995 --nesterov --mu 7 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 8 --warmup-steps 4000 --uda-steps 5000 --student-wait-steps 3000 --teacher-dropout 0.2 --student-dropout 0.2 --amp
```

Train the model by 10000 labeled data of CIFAR-100 dataset by using DistributedDataParallel:
```
python -m torch.distributed.launch --nproc_per_node 4 main.py --seed 42 --name HBS3.5k --expand-labels --dataset hbs --num-classes 9 --total-steps 100000 --eval-step 1000 --randaug 2 16 --batch-size 18 --teacher_lr 0.025 --student_lr 0.025 --weight-decay 5e-4 --ema 0.995 --nesterov --mu 7 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 8 --warmup-steps 4000 --uda-steps 5000 --student-wait-steps 3000 --teacher-dropout 0.2 --student-dropout 0.2 --amp 
```

Monitoring training progress
```
tensorboard --logdir results
```

## Requirements
- python 3.6+
- torch 1.7+
- torchvision 0.8+
- tensorboard
- numpy
- tqdm
