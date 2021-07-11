import logging
import math
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib2 import Path
from torch._C import Size

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

from augmentation import RandAugment

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

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
        transforms.Resize((int(args.resize * 1.5), int(args.resize * 1.5))),
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
    labeled_dataset = HBSLabeled('/home/amax/Algorithm/MPL-pytorch/HBSLabeled')
    train_unlabeled_dataset = HBSUnlabeled(
        '/home/amax/Algorithm/MPL-pytorch/HBSUnlabeled', transform_MPL)
    len_labled = len(labeled_dataset)
    del labeled_dataset
    len_train_labeled = int(len_labled * 0.95)
    np.random.seed(args.seed)
    idx = np.random.permutation(len_labled)
    train_labeled_idxs = idx[:len_train_labeled]
    val_labeled_idxs = idx[len_train_labeled:]

    train_labeled_dataset = HBSLabeled(
        '/home/amax/Algorithm/MPL-pytorch/HBSLabeled', train_labeled_idxs,
        transform_labeled)
    test_dataset = HBSLabeled('/home/amax/Algorithm/MPL-pytorch/HBSLabeled',
                              val_labeled_idxs, transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar10(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(args.data_path, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)
    # train_labeled_idxs, train_unlabeled_idxs = x_u_split_test(args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(args.data_path,
                                       train_labeled_idxs,
                                       train=True,
                                       transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(args.data_path,
                                         train_unlabeled_idxs,
                                         train=True,
                                         transform=TransformMPL(
                                             args,
                                             mean=cifar10_mean,
                                             std=cifar10_std))

    test_dataset = datasets.CIFAR10(args.data_path,
                                    train=False,
                                    transform=transform_val,
                                    download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])

    base_dataset = datasets.CIFAR100(args.data_path, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(args.data_path,
                                        train_labeled_idxs,
                                        train=True,
                                        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(args.data_path,
                                          train_unlabeled_idxs,
                                          train=True,
                                          transform=TransformMPL(
                                              args,
                                              mean=cifar100_mean,
                                              std=cifar100_std))

    test_dataset = datasets.CIFAR100(args.data_path,
                                     train=False,
                                     transform=transform_val,
                                     download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all training data
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(args.batch_size * args.eval_step /
                                 args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


def x_u_split_test(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[label_per_class:])
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(args.batch_size * args.eval_step /
                                 args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])

    np.random.shuffle(labeled_idx)
    np.random.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx


class TransformMPL(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default

        self.ori = transforms.Compose([
            transforms.Resize(
                (int(args.resize * 1.5), int(args.resize * 1.5))),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(args.resize, args.resize),
                                  padding=int(args.resize * 0.125),
                                  padding_mode='reflect')
        ])
        self.aug = transforms.Compose([
            transforms.Resize(
                (int(args.resize * 1.5), int(args.resize * 1.5))),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(args.resize, args.resize),
                                  padding=int(args.resize * 0.125),
                                  padding_mode='reflect'),
            RandAugment(n=n, m=m)
        ])
        self.normalize = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self,
                 root,
                 indexs,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super().__init__(root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self,
                 root,
                 indexs,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super().__init__(root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


DATASET_GETTERS = {
    'cifar10': get_cifar10,
    'cifar100': get_cifar100,
    'hbs': get_hbs
}
