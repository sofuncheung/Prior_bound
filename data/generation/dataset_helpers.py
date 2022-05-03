import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision as tv

from .experiment_config import Config, DatasetType, HParams


def get_dataloaders(hparams: HParams, config: Config, device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if hparams.dataset_type == DatasetType.CIFAR10:
        dataset = CIFAR10
    elif hparams.dataset_type == DatasetType.SVHN:
        dataset = SVHN
    elif hparams.dataset_type == DatasetType.CIFAR10_binary:
        dataset = CIFAR10_binary
    elif hparams.dataset_type == DatasetType.SVHN_binary:
        dataset = SVHN_binary
    elif hparams.dataset_type == DatasetType.MNIST_binary:
        dataset = MNIST_binary
    elif hparams.dataset_type == DatasetType.FashionMNIST_binary:
        dataset = FashionMNIST_binary
    elif hparams.dataset_type == DatasetType.KMNIST_binary:
        dataset = KMNIST_binary
    elif hparams.dataset_type == DatasetType.EMNIST_binary:
        dataset = EMNIST_binary
    elif hparams.dataset_type == DatasetType.PCAM:
        dataset = PCAM

    else:
        raise KeyError

    if (hparams.dataset_type == DatasetType.SVHN or
        hparams.dataset_type == DatasetType.SVHN_binary):
        train_key = {'split': 'train'}
        test_key = {'split': 'test'}
    else:
        train_key = {'train': True}
        test_key = {'train': False}

    '''
    train_key = {'split': 'train'} if hparams.dataset_type == DatasetType.SVHN else {'train': True}
    test_key = {'split': 'test'} if hparams.dataset_type == DatasetType.SVHN else {'train': False}
    '''
    if hparams.dataset_type == DatasetType.EMNIST_binary:
        # Using EMNIST-Digits only
        train = dataset(hparams, config, device, 'digits',
                download=True, **train_key)
        test = dataset(hparams, config, device, 'digits',
                download=True, **test_key)
    else:
        train = dataset(hparams, config, device, download=True, **train_key)
        test = dataset(hparams, config, device, download=True, **test_key)

    train_loader = DataLoader(train, batch_size=hparams.batch_size, shuffle=True, num_workers=0)
    train_eval_loader = DataLoader(train, batch_size=500, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=500, shuffle=False, num_workers=0)
    return train_loader, train_eval_loader, test_loader


def process_data(hparams: HParams, data_np: np.ndarray, targets_np: np.ndarray, device:
        torch.device, train: bool, binary: bool = False):
    # Scale data to [0,1] floats
    data_np = data_np / 255

    if hparams.center_data == True:
        # Centering data
        data_np = (data_np - data_np.mean(axis=(0,1,2))) / data_np.std(axis=(0,1,2))

    # NHWC -> NCHW
    data_np = data_np.transpose((0,3,1,2))

    # Binarize targets
    if binary:
        print('Using binary data...')
        targets_np = np.where(targets_np < 5, 0, 1)

    # Numpy -> Torch
    data = torch.tensor(data_np, dtype=torch.float32) # Memory checkpoint
    targets = torch.tensor(targets_np, dtype=torch.float32)

    # Resize dataset
    dataset_size, offset = (hparams.train_dataset_size, 0) if train else (hparams.test_dataset_size, 1)
    if dataset_size is not None:
        rng = np.random.RandomState(hparams.data_seed + offset) if (hparams.data_seed is not None) else np.random
        indices = rng.choice(len(data), dataset_size, replace=False)
        indices = torch.from_numpy(indices)
        data = torch.index_select(data, 0, indices)
        targets = torch.index_select(targets, 0, indices)

    # Put both data and targets on GPU in advance
    return data.to(device), targets.to(device)


# https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
# We need to keep the class name the same as base class methods rely on it
class CIFAR10(tv.datasets.CIFAR10):
    def __init__(self, hparams: HParams, config: Config, device: torch.device, *args, **kwargs):
        super().__init__(config.data_dir, *args, **kwargs)
        self.data, self.targets = process_data(hparams, self.data, np.array(self.targets), device, self.train)

    # Don't convert to PIL like torchvision default
    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class SVHN(tv.datasets.SVHN):
    def __init__(self, hparams: HParams, config: Config, device: torch.device, *args, **kwargs):
        super().__init__(config.data_dir, *args, **kwargs)
        self.data = self.data.transpose((0, 2, 3, 1)) # NCHW -> NHWC (SVHN)
        self.data, self.labels = process_data(hparams, self.data, self.labels, device, self.split == 'train')

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class CIFAR10_binary(tv.datasets.CIFAR10):
    def __init__(self, hparams: HParams, config: Config, device: torch.device, *args, **kwargs):
        super().__init__(config.data_dir, *args, **kwargs)
        self.data, self.targets = process_data(hparams, self.data, np.array(self.targets),
                device, self.train, binary = True)

    # Don't convert to PIL like torchvision default
    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class SVHN_binary(tv.datasets.SVHN):
    def __init__(self, hparams: HParams, config: Config, device: torch.device, *args, **kwargs):
        super().__init__(config.data_dir, *args, **kwargs)
        self.data = self.data.transpose((0, 2, 3, 1)) # NCHW -> NHWC (SVHN)
        self.data, self.labels = process_data(hparams, self.data, self.labels,
                device, self.split == 'train', binary=True)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class MNIST_binary(tv.datasets.MNIST):
    def __init__(self, hparams: HParams, config: Config, device: torch.device, *args, **kwargs):
        super().__init__(config.data_dir, *args, **kwargs)
        self.data = np.expand_dims(self.data, -1) # NHW -> NHWC
        self.data, self.targets = process_data(hparams, self.data, np.array(self.targets),
                device, self.train, binary=True)
        #print(self.data.shape)
        #print(self.targets.shape)
        #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class FashionMNIST_binary(tv.datasets.FashionMNIST):
    def __init__(self, hparams: HParams, config: Config, device: torch.device, *args, **kwargs):
        super().__init__(config.data_dir, *args, **kwargs)
        self.data = np.expand_dims(self.data, -1) # NHW -> NHWC
        self.data, self.targets = process_data(hparams, self.data, np.array(self.targets),
                device, self.train, binary=True)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class KMNIST_binary(tv.datasets.KMNIST):
    def __init__(self, hparams: HParams, config: Config, device: torch.device, *args, **kwargs):
        super().__init__(config.data_dir, *args, **kwargs)
        self.data = np.expand_dims(self.data, -1) # NHW -> NHWC
        self.data, self.targets = process_data(hparams, self.data, np.array(self.targets),
                device, self.train, binary=True)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class EMNIST_binary(tv.datasets.EMNIST):
    def __init__(self, hparams: HParams, config: Config, device: torch.device, *args, **kwargs):
        super().__init__(config.data_dir, *args, **kwargs)
        self.data = np.expand_dims(self.data, -1) # NHW -> NHWC
        self.data, self.targets = process_data(hparams, self.data, np.array(self.targets),
                device, self.train, binary=True)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

'''
class PCAM(tv.datasets.PCAM):
    def __init__(self, hparams: HParams, config: Config, device: torch.device, *args, **kwargs):
        super().__init__(config.data_dir, *args, **kwargs)
        images_file = self._FILES[self._split]["images"][0]
        with self.h5py.File(self._base_folder / images_file) as images_data:
            self.data = images_data["x"] # NHWC 0-255
        targets_file = self._FILES[self._split]["targets"][0]
        with self.h5py.File(self._base_folder / targets_file) as targets_data:
            self.targets = int(np.flatten(targets_data["y"]))
            # targets_data["y"] is of dimension (N, 1, 1, 1)
        self.data, self.targets = process_data(hparams, self.data, self.targets,
                device, self.split == 'train', binary=False)
        # Note: PCAM is actually binary but here binary=False.
        # That's because here binary means "binaring 10-class to 2-class",
        # which PCAM doesn't need.

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
'''
