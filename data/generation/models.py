from collections import OrderedDict
import torch
from torch import Tensor
import math
import torch.nn as nn
import torch.nn.functional as F

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
        # x = self.relu(x)
        # For single logit output need to remove the final relu

        x = self.avgpool(x)

        return x.squeeze().unsqueeze(-1)
        # to match the shape of FCN outputs when there's only single output logit
        # (Although uncessary)

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

        self.input_dim = calculate_production(self.dataset_type.D)
        self.width_tuple = width_tuple
        self.number_layers = len(width_tuple) # number of hidden layers
        self.model = [nn.Flatten()]
        self.model.append(nn.Linear(self.input_dim, width_tuple[0]))
        self.model.append(nn.ReLU(inplace=True))
        for i in range(len(width_tuple) - 1):
            self.model.append(nn.Linear(width_tuple[i], width_tuple[i+1]))
            self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.Linear(width_tuple[-1], self.dataset_type.K))
        self.model = nn.Sequential(*self.model)
        self.apply(self.he_init)
        self.init_w_b = self.get_weights_and_biases_data(self.model)

    def he_init(self, module):
        r'''
        He-normal initialization for GP volume calculation.
        The function should be used in conjuction with 'net.apply'
        '''
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight,
                    mode='fan_in', # GP only involves feed-forward process
                    nonlinearity='relu') # so gain = sqrt(2)
            if (not (module.bias is None)):
                nn.init.normal_(module.bias, std=0.1)

    @staticmethod
    def get_weights_and_biases_data(model):
        return [p.data for name, p in model.named_parameters()]


    def forward(self, x):
        return self.model(x)


class FCN_binary_test(ExperimentBaseModel):
    def __init__(self, width_tuple: list, dataset_type: DatasetType) -> None:
        super().__init__(dataset_type)

        self.input_dim = calculate_production(self.dataset_type.D)
        self.width_tuple = width_tuple
        self.number_layers = len(width_tuple) # number of hidden layers
        self.model = [nn.Flatten()]
        self.model.append(nn.Linear(self.input_dim, width_tuple[0]))
        self.model.append(nn.ReLU(inplace=True))
        for i in range(len(width_tuple) - 1):
            self.model.append(nn.Linear(width_tuple[i], width_tuple[i+1]))
            self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.Linear(width_tuple[-1], 2))
        self.model = nn.Sequential(*self.model)
        self.apply(self.he_init)
        self.init_w_b = self.get_weights_and_biases_data(self.model)

    def he_init(self, module):
        r'''
        He-normal initialization for GP volume calculation.
        The function should be used in conjuction with 'net.apply'
        '''
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight,
                    mode='fan_in', # GP only involves feed-forward process
                    nonlinearity='relu') # so gain = sqrt(2)
            if (not (module.bias is None)):
                nn.init.normal_(module.bias, std=0.1)

    @staticmethod
    def get_weights_and_biases_data(model):
        return [p.data for name, p in model.named_parameters()]

    def forward(self, x):
        x = self.model(x)
        return x[:,1] - x[:,0]

    # If using this model for empirical kernal calculation, the final result needs to be 
    # divided by 2. This can be easily seen by Cov( y1-y2, y1'-y2' ) = 2 x Cov( y1, y1' ) = 2 x Cov(y2, y2')


class FCN_scale_ignorant(ExperimentBaseModel):
    """
    FCN with weights initialized regardless of fan_in.
    The inituition is SGD can still train this non-conventionally
    initialized network, but the corresponding NNGP will perform
    badly, because the per-unit l2 norm is not bounded anymore.
    """
    def __init__(self, width_tuple: list,
            dataset_type: DatasetType,
            SI_w_std: float) -> None:
        super().__init__(dataset_type)

        input_dim = calculate_production(self.dataset_type.D)

        self.number_layers = len(width_tuple) # number of hidden layers
        self.model = [nn.Flatten()]
        self.model.append(nn.Linear(input_dim, width_tuple[0]))
        self.model.append(nn.ReLU(inplace=True))
        for i in range(len(width_tuple) - 1):
            self.model.append(nn.Linear(width_tuple[i], width_tuple[i+1]))
            self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.Linear(width_tuple[-1], self.dataset_type.K))
        self.model = nn.Sequential(*self.model)

        self.SI_w_std = SI_w_std
        self.apply(self.SI_init)

    def SI_init(self, module):
        r'''
        Trivial initialization which every weights are initialized regardless
        of fan_in.

        The function should be used in conjuction with 'net.apply'
        '''
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.SI_w_std)
            if (not (module.bias is None)):
                nn.init.normal_(module.bias, std=0.1)

    def forward(self, x):
        return self.model(x)


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

        # this asymmetric padding layer is only needed beford an intermediate
        # pooling layer when the feature map dims is odd.


        self.model = [nn.Conv2d(self.dataset_type.D[0],
            width_tuple[0], self.filter_sizes[0], padding=self.padding[0])]
        self.model.append(nn.ReLU(inplace=True))
        if intermediate_pooling_type != None:
            if cal_image_dim(self.dataset_type.D[1], 1) % 2 == 1:
                self.model.append(nn.ZeroPad2d((0,1,0,1))) # feature has odd dim, needs asym padding
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
                    self.model.append(nn.ZeroPad2d((0,1,0,1)))
                if intermediate_pooling_type == "avg":
                    self.model.append(nn.AvgPool2d(2, count_include_pad=False))
                elif intermediate_pooling_type == "max":
                    self.model.append(nn.MaxPool2d(2))
        #Global pooling
        if pooling == "avg":
            self.model.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif pooling == "max":
            self.model.append(nn.AdaptiveMaxPool2d((1, 1)))

        self.model.append(nn.Flatten())
        self.model.append(nn.Linear(width_tuple[-1], self.dataset_type.K))
        self.model = nn.Sequential(*self.model)
        self.apply(self.he_init)

    def he_init(self, module):
        r'''
        He-normal initialization for GP volume calculation.
        The function should be used in conjuction with 'net.apply'
        (21 Feb 2023) Also note that in Guillermo's CNN vs layers
        experiments, sigmab = 0.1.
        '''
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight,
                    mode='fan_in', # GP only involves feed-forward process
                    nonlinearity='relu') # so gain = sqrt(2)
            if (not (module.bias is None)):
                nn.init.normal_(module.bias, std=0.1)

    def forward(self, x):
        return self.model(x)


# Following are manual constraction of Resnets.
# To facilitate using _reparam(model) in measure.py, all conv layers
# have biases.

class BaseBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dim_change=None):
        super(BaseBlock, self).__init__()
        # Declare convolutional layers with batch norms
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, )
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, ),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dim_change=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, )
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.dim_change = dim_change

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, ),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(ExperimentBaseModel):
    def __init__(self, block, num_blocks, dataset_type):
        super(ResNet, self).__init__(dataset_type)
        self.in_planes = 64

        self.conv1 = nn.Conv2d(dataset_type.D[0], 64, kernel_size=3,
                               stride=1, padding=1, )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, dataset_type.K)
        self.apply(self.he_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @staticmethod
    def he_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,
                    mode='fan_in', # GP only involves feed-forward process
                    nonlinearity='relu') # so gain = sqrt(2)
            if (not (m.bias is None)):
                nn.init.normal_(m.bias, mean=0.0, std=0.1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_pop_fc(ResNet):
    r'''
    Last fc layer poped. For emperical kernal calculation.
    (or possibly useful for transfer learning as well)
    '''
    def __init__(self, block, num_blocks, dataset_type):
        super().__init__(block, num_blocks, dataset_type)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def ResNet50(dataset_type):
    return ResNet(Bottleneck, [3, 4, 6, 3], dataset_type)


def ResNet_pop_fc_50(dataset_type):
    return ResNet_pop_fc(Bottleneck, [3, 4, 6, 3], dataset_type)


"""
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
"""

# Implementation of Densenet

# The pytorch implementation has evolved a lot from the earlier version to accommodate
# the growing-complexity pytorch ecosystem. But here what we really need is just the
# basic Densenet structure. So the implementation below will be minimal.
# Also, as in ResNet, here Conv layers all have biases.

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1,))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,)),
        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        "Bottleneck function"
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input):
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(ExperimentBaseModel):
    def __init__(self, dataset_type, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super().__init__(dataset_type)

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(dataset_type.D[0],
                num_init_features, kernel_size=7, stride=2,
                padding=5)) if dataset_type.D[-1] < 32 else
            ('conv0', nn.Conv2d(dataset_type.D[0],
                num_init_features, kernel_size=7, stride=2,
                padding=3)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Add multiple denseblocks based on config
        # for densenet-121 config: [6,12,24,16]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                # add transition layer between denseblocks to
                # downsample
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, dataset_type.K)

        self.apply(self.he_init)

    @staticmethod
    def he_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,
                    mode='fan_in', # GP only involves feed-forward process
                    nonlinearity='relu') # so gain = sqrt(2)
            if (not (m.bias is None)):
                nn.init.normal_(m.bias, mean=0.0, std=0.1)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class DenseNet_fc_popped(DenseNet):

    def __init__(self, dataset_type, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):
        super().__init__(dataset_type, growth_rate=32, block_config=(6, 12, 24, 16),
                num_init_features=64, bn_size=4, drop_rate=0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out


def DenseNet121(dataset_type):
    return(DenseNet(dataset_type, 32, (6, 12, 24, 16), 64))


def DenseNet121_fc_popped(dataset_type):
    return(DenseNet_fc_popped(dataset_type, 32, (6, 12, 24, 16), 64))













