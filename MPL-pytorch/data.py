import logging
import math
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib2 import Path

from torch.utils.data import Dataset
from torchvision import transforms

from augmentation import RandAugment

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# Base wrapper for Labeled Huawei Banner Style Dataset
class HBSLabeled(Dataset):
    def __init__(self, data_root, idxs=None, transform=None):
        data_root = Path(data_root)
        self.img_dir = data_root / 'image'
        label_path = data_root / 'labels.csv'
        self.labels = pd.read_csv(label_path)
        if idxs is not None:
            self.labels = self.labels.iloc[idxs, :]
        self.transform = transform

    def __getitem__(self, index):
        # read i-th row of a panda dataframe
        label_info = self.labels.iloc[index, :]
        img_path = self.img_dir / label_info['img_file_name']
        label = label_info['label']
        img = Image.open(str(img_path)).convert('RGB')
        return self.transform(img), label

    def __len__(self):
        return len(self.labels)


# Base wrapper for Unlabeled Huawei Banner Style Dataset
class HBSUnlabeled(Dataset):
    def __init__(self, data_root, transform=None):
        data_root = Path(data_root)
        self.img_paths = tuple(data_root.iterdir())
        self.transform = transform

    def __getitem__(self, index):
        # read i-th row of a panda dataframe
        img_path = self.img_paths[index]
        img = Image.open(str(img_path)).convert('RGB')
        return self.transform(img), -1

    def __len__(self):
        return len(self.img_paths)


def get_hbs(args):
    transform_labeled = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    transform_val = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    transform_MPL = TransformMPL(args, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    labeled_dataset = HBSLabeled('/home/bz/Algorithm/MPL-pytorch/HBSLabeled')
    train_unlabeled_dataset = HBSUnlabeled(
        '/home/bz/Algorithm/MPL-pytorch/HBSUnlabeled', transform_MPL)
    len_labled = len(labeled_dataset)
    del labeled_dataset
    len_train_labeled = int(len_labled * 0.925)
    np.random.seed(args.seed)
    idx = np.random.permutation(len_labled)
    train_labeled_idxs = idx[:len_train_labeled]
    val_labeled_idxs = idx[len_train_labeled:]

    train_labeled_dataset = HBSLabeled(
        '/home/bz/Algorithm/MPL-pytorch/HBSLabeled', train_labeled_idxs,
        transform_labeled)
    test_dataset = HBSLabeled('/home/bz/Algorithm/MPL-pytorch/HBSLabeled',
                              val_labeled_idxs, transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class TransformMPL(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default

        self.ori = transforms.Compose([
            transforms.Resize((args.resize, args.resize)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(args.resize, args.resize),
                                  padding=int(args.resize * 0.125))
        ])
        self.aug = transforms.Compose([
            transforms.Resize((args.resize, args.resize)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(args.resize, args.resize),
                                  padding=int(args.resize * 0.125)),
            RandAugment(n=n, m=m)
        ])
        self.normalize = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)


DATASET_GETTERS = {'hbs': get_hbs}
