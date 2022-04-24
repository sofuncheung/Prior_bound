from torch import Tensor
import math
import torch.nn as nn

from .experiment_config import DatasetType


class ExperimentBaseModel(nn.Module):
    def __init__(self, dataset_type: DatasetType):
        super().__init__()
        self.dataset_type = dataset_type

    def forward(self, x) -> Tensor:
        raise NotImplementedError


class NiNBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x


class NiN(ExperimentBaseModel):
    def __init__(self, depth: int, width: int, base_width: int, dataset_type: DatasetType) -> None:
        super().__init__(dataset_type)

        self.base_width = base_width

        blocks = []
        blocks.append(NiNBlock(self.dataset_type.D[0], self.base_width*width))
        for _ in range(depth-1):
            blocks.append(NiNBlock(self.base_width *
                          width, self.base_width*width))
        self.blocks = nn.Sequential(*blocks)

        self.conv = nn.Conv2d(self.base_width*width,
                              self.dataset_type.K, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(self.dataset_type.K)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.blocks(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)

        return x.squeeze()

class NiN_binary(ExperimentBaseModel):
    def __init__(self, depth: int, width: int, base_width: int, dataset_type: DatasetType) -> None:
        super().__init__(dataset_type)

        assert self.dataset_type.K == 2, "NiN_binary can only be used with two logits"

        self.base_width = base_width

        blocks = []
        blocks.append(NiNBlock(self.dataset_type.D[0], self.base_width*width))
        for _ in range(depth-1):
            blocks.append(NiNBlock(self.base_width *
                          width, self.base_width*width))
        self.blocks = nn.Sequential(*blocks)

        self.conv = nn.Conv2d(self.base_width*width,
                              self.dataset_type.K, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(self.dataset_type.K)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.blocks(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.squeeze()

        return x[:,1] - x[:,0]


def calculate_production(t: tuple) -> int:
    p = 1
    for i in t:
        p = p * i

    return p


class FCN(ExperimentBaseModel):
    def __init__(self, width_tuple: list, dataset_type: DatasetType) -> None:
        super().__init__(dataset_type)

        input_dim = calculate_production(self.dataset_type.D)

        self.model = [nn.Flatten()]
        self.model.append(nn.Linear(input_dim, width_tuple[0]))
        self.model.append(nn.ReLU(inplace=True))
        for i in range(len(width_tuple) - 1):
            self.model.append(nn.Linear(width_tuple[i], width_tuple[i+1]))
            self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.Linear(width_tuple[-1], self.dataset_type.K))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class FCN_binary(ExperimentBaseModel):
    def __init__(self, width_tuple: list, dataset_type: DatasetType) -> None:
        super().__init__(dataset_type)

        input_dim = calculate_production(self.dataset_type.D)

        self.model = [nn.Flatten()]
        self.model.append(nn.Linear(input_dim, width_tuple[0]))
        self.model.append(nn.ReLU(inplace=True))
        for i in range(len(width_tuple) - 1):
            self.model.append(nn.Linear(width_tuple[i], width_tuple[i+1]))
            self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.Linear(width_tuple[-1], self.dataset_type.K))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = self.model(x)
        return x[:,1] - x[:,0]


def cal_image_dim(input_dim: int, l: int) -> int:
    """
    Because we want this CNN to be the same as in Guillermo's paper,
    which use "SAME" padding in intermediate pooling layers,
    we need this function to calculate the image dimensions after each Conv2D
    layer (before intermediate pooling layer) and pad the image (asymmetrically) accordingly.

    Args:
        input_dim: the original dim of squared images. e.g. 32 for CIFAR, 28 for MNIST
        l: the index of the Conv2D layer, starting at 1.
    """

    # assert l <= 4, "for CIFAR and MNIST, with intermediate pooling l must <= 4!!!"

    h = input_dim
    if l == 0:
        return h
    if l == 1:
        h = h - 4
    else:
        h = h - 4
        for i in range((l-1) // 2):
            h = math.ceil(h / 2) # pooling (stride = 2) + Conv2D (same)
            h = math.ceil(h / 2) - 4 # pooling (stride = 2) + Conv2D (valid, kernel=(5,5))
        for i in range((l-1) % 2):
            h = math.ceil(h / 2)
    if h < 1:
        raise ValueError("Feature map dim < 1!!! Downsampling is too much...")

    return h


class CNN(ExperimentBaseModel):
    def __init__(self, width_tuple: list,
            intermediate_pooling_type: str,
            pooling: str,
            dataset_type: DatasetType) -> None:
        super().__init__(dataset_type)

        self.L = len(width_tuple) # number of hidden layers
        self.filter_sizes = [[5,5], [2,2]] * (self.L // 2) + [[5,5]] * (self.L % 2)
        self.padding = ["valid", "same"] * (self.L // 2) + ["valid"] * (self.L % 2)

        self.pooling_same_padding = nn.ZeroPad2d((0,1,0,1))
        # this asymmetric padding layer is only needed beford an intermediate
        # pooling layer when the feature map dims is odd.


        self.model = [nn.Conv2d(self.dataset_type.D[0],
            width_tuple[0], self.filter_sizes[0], padding=self.padding[0])]
        self.model.append(nn.ReLU(inplace=True))
        if intermediate_pooling_type != None:
            if cal_image_dim(self.dataset_type.D[1], 1) % 2 == 1:
                self.model.append(self.pooling_same_padding) # feature has odd dim, needs asym padding
            if intermediate_pooling_type == "avg":
                self.model.append(nn.AvgPool2d(2, count_include_pad=False))
            elif intermediate_pooling_type == "max":
                self.model.append(nn.MaxPool2d(2))

        for i in range(self.L - 1):
            self.model.append(nn.Conv2d(width_tuple[i], width_tuple[i+1],
                                        self.filter_sizes[i+1], padding=self.padding[i+1]
                                       )
                             )
            self.model.append(nn.ReLU(inplace=True))
            if intermediate_pooling_type != None:
                if cal_image_dim(self.dataset_type.D[1], i+2) % 2 == 1:
                    self.model.append(self.pooling_same_padding)
                if intermediate_pooling_type == "avg":
                    self.model.append(nn.AvgPool2d(2, count_include_pad=False))
                elif intermediate_pooling_type == "max":
                    self.model.append(nn.MaxPool2d(2))
        # Global pooling
        if pooling == "avg":
            self.model.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif pooling == "max":
            self.model.append(nn.AdaptiveMaxPool2d((1, 1)))

        self.model.append(nn.Flatten())
        self.model.append(nn.Linear(width_tuple[-1], self.dataset_type.K))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class CNN_binary(CNN):
    def __init__(self, width_tuple: list,
            intermediate_pooling_type: str,
            pooling: str,
            dataset_type: DatasetType) -> None:
        super().__init__(width_tuple,
                intermediate_pooling_type,
                pooling,
                dataset_type)

    def forward(self, x):
        x = self.model(x)
        return x[:,1] - x[:,0]



