import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision as tv

from .experiment_config import Config, DatasetType, HParams, LossType


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

        # For MSE loss using +1 -1 labels
        if hparams.loss == LossType.MSE:
            targets_np = np.where(targets_np == 0, -1, 1)


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

    # Label corruption (change labels in a portion of the training set to random labels)
    # Note on 15 Aug: fixed bug where it used to corrupt the labels in the test set too.
    if train and (hparams.label_corruption is not None):
        corruption_size, offset_corruption = (int(hparams.label_corruption*hparams.train_dataset_size), 2)
        rng_label_corruption = np.random.RandomState(
                hparams.data_seed + offset_corruption) if (hparams.data_seed is not None) else np.random
        corrupt_indices = rng_label_corruption.choice(hparams.train_dataset_size,
                corruption_size, replace=False) # Use same rng as data selection
        corrupt_indices = torch.from_numpy(corrupt_indices)
        for corrupt_index in corrupt_indices:
            targets[corrupt_index] = torch.tensor(rng_label_corruption.choice([0, 1]),
                    dtype=torch.float32)

    # Attack set (add additional attack set in which real labels are flipped)
    if hparams.attack_dataset_size is not None:
        raise NotImplementedError

    # Shuffle pixels per image (which is different from label corruption as a way of
    # increasing data complexity)
    if (train and hparams.shuffle_pixel_per_image_train == True) or (
            (not train) and hparams.shuffle_pixel_per_image_test == True):
        '''
        data = torch.tensor([[[[1,2,3],[4,5,6],[7,8,9]],
            [[11,12,13],[14,15,16],[17,18,19]], [[21,22,23],[24,25,26],[27,28,29]]],
            [[[101,102,103],[104,105,106],[107,108,109]],
                [[111,112,113],[114,115,116],[117,118,119]], [[121,122,123],[124,125,126],[127,128,129]]
            ]])
        '''
        offset_shuffle_pixel_per_image = 3 if train else 4
        rng_shuffle_pixel_per_image = np.random.RandomState(
                hparams.data_seed + offset_shuffle_pixel_per_image ) if (hparams.data_seed is not None) else np.random
        ori_shape_per_image = data[0].shape
        if hparams.shuffle_pixel_per_image_mode == "intra-channel-tied":
            for row_idx in range(len(data)):
                idx = np.arange(ori_shape_per_image[-1] * ori_shape_per_image[-2])
                rng_shuffle_pixel_per_image.shuffle(idx)
                data[row_idx] = data[row_idx].view(ori_shape_per_image[0],-1).transpose(1,0)[idx].transpose(1,0).view(ori_shape_per_image)

        elif hparams.shuffle_pixel_per_image_mode == "intra-channel":
            for row_idx in range(len(data)):
                for channel_idx in range(ori_shape_per_image[0]):
                    idx = np.arange(ori_shape_per_image[-1] * ori_shape_per_image[-2])
                    rng_shuffle_pixel_per_image.shuffle(idx)
                    data[row_idx][channel_idx] = data[row_idx][channel_idx].view(
                            -1)[idx].view(data[row_idx][channel_idx].shape)

        elif hparams.shuffle_pixel_per_image_mode == "inter-channel":
            for row_idx in range(len(data)):
                idx = np.arange(np.prod(ori_shape_per_image))
                rng_shuffle_pixel_per_image.shuffle(idx)
                data[row_idx] = data[row_idx].view(-1)[idx].view(data[row_idx].shape)

    # Shuffle pixels but only choose the random permutation once and apply to
    # all images in both traning set and test set (As in paper "MEASURING THE 
    # INTRINSIC DIMENSION OF OBJECTIVE LANDSCAPES")
    if hparams.shuffle_pixel_all_once_train == True:
        offset_shuffle_pixel_all_once = 5
        rng_shuffle_pixel_all_once = np.random.RandomState(
                hparams.data_seed + offset_shuffle_pixel_all_once ) if (
                        hparams.data_seed is not None) else np.random
        ori_shape_dataset = data.shape
        if hparams.shuffle_pixel_all_once_mode == "intra-channel-tied":
            idx = np.arange(ori_shape_dataset[-1] * ori_shape_dataset[-2])
            rng_shuffle_pixel_all_once.shuffle(idx)
            if (train or hparams.shuffle_pixel_all_once_test):
                data = data.view(ori_shape_dataset[0],ori_shape_dataset[1],-1).transpose(
                        0,2)[idx].transpose(0,2).view(ori_shape_dataset)
        elif hparams.shuffle_pixel_all_once_mode == "intra-channel":
            for channel_idx in range(ori_shape_dataset[1]):
                idx = np.arange(ori_shape_dataset[-1] * ori_shape_dataset[-2])
                rng_shuffle_pixel_all_once.shuffle(idx)
                if (train or hparams.shuffle_pixel_all_once_test):
                    data[:,channel_idx,:,:] = data[:,channel_idx,:,:].view(
                            ori_shape_dataset[0],-1).transpose(0,1)[idx].transpose(
                                    0,1).view(data[:,channel_idx,:,:].shape)
        elif hparams.shuffle_pixel_all_once_mode == "inter-channel":
            idx = np.arange(np.prod(ori_shape_dataset[1:]))
            rng_shuffle_pixel_all_once.shuffle(idx)
            if (train or hparams.shuffle_pixel_all_once_test):
                data = data.view(ori_shape_dataset[0],-1).transpose(
                        0,1)[idx].transpose(0,1).view(ori_shape_dataset)


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
