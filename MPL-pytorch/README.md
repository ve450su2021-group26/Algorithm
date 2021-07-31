# Meta Pseudo Labels
This is an unofficial PyTorch implementation of [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580) specified on our task .
The original unofficial PyTorch implementation is [here](https://github.com/kekmodel/MPL-pytorch)
The official Tensorflow implementation is [here](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels).


## Usage

Train the model on HBS data set

```
python main.py --seed 42 --name HBS15k --expand-labels --dataset hbs --num-classes 9 --total-steps 150000 --eval-step 1000 --randaug 2 12 --batch-size 18 --teacher_lr 0.075 --student_lr 0.05 --weight-decay 1e-4 --ema 0.998 --nesterov --mu 7 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 14 --warmup-steps 5000 --uda-steps 25000 --student-wait-steps 7000 --teacher-dropout 0.1 --student-dropout 0.1 --amp --resize 224
```

Train the model by 10000 labeled data of CIFAR-100 dataset by using DistributedDataParallel:
```
python -m torch.distributed.launch --nproc_per_node 4 main.py --seed 42 --name HBS15k --expand-labels --dataset hbs --num-classes 9 --total-steps 150000 --eval-step 1000 --randaug 2 12 --batch-size 18 --teacher_lr 0.075 --student_lr 0.05 --weight-decay 1e-4 --ema 0.998 --nesterov --mu 7 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 14 --warmup-steps 5000 --uda-steps 25000 --student-wait-steps 7000 --teacher-dropout 0.1 --student-dropout 0.1 --amp --resize 224
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
