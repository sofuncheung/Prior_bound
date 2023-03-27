from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
import hashlib
from pathlib import Path
import time
from typing import Dict, List, NamedTuple, Optional, Tuple


class DatasetType(Enum):
    CIFAR10 = (1, (3, 32, 32), 10)
    SVHN = (2, (3, 32, 32), 10)
    CIFAR10_binary = (3, (3, 32, 32), 1)
    SVHN_binary = (4, (3, 32, 32), 1)
    MNIST_binary = (5, (1, 28, 28), 1)
    FashionMNIST_binary = (6, (1, 28, 28), 1)
    KMNIST_binary = (7, (1, 28, 28), 1)
    EMNIST_binary = (8, (1, 28, 28), 1)
    PCAM = (9, (3, 96, 96), 1)

    def __init__(self, id: int, image_shape: Tuple[int, int, int], num_logits: int):
        self.D = image_shape
        self.K = num_logits


class DatasetSubsetType(IntEnum):
    TRAIN = 0
    TEST = 1


class ModelType(Enum):
    NiN = 0
    FCN = 1
    CNN = 2
    FCN_SI = 3
    RESNET50 = 4
    DENSENET121 = 5
    DENSENET_WO_BIAS_121 = 6

class ComplexityType(Enum):
    # GP based Measures
    PRIOR = 1
    MAR_LIK = 2
    PRIOR_MC = 3
    PRIOR_ELBO = 4
    MAR_LIK_MC = 5
    MAR_LIK_ELBO = 6
    # PAC-Bayes bound optimization
    BPAC_OPT = 7
    B_RE_OPTIMAL = 8
    MEAN_TRAIN_ACC_STOCH = 9
    MEAN_TEST_ACC_STOCH = 10
    TRAIN_ACC_DET = 11
    TEST_ACC_DET = 12
    # Path-norm bound
    PATH_NORM_BOUND = 13
    # Measures from Fantastic Generalization Measures (equation numbers)
    PARAMS = 20
    INVERSE_MARGIN = 22
    LOG_SPEC_INIT_MAIN = 29
    LOG_SPEC_ORIG_MAIN = 30
    LOG_PROD_OF_SPEC_OVER_MARGIN = 31
    LOG_PROD_OF_SPEC = 32
    FRO_OVER_SPEC = 33
    LOG_SUM_OF_SPEC_OVER_MARGIN = 34
    LOG_SUM_OF_SPEC = 35
    LOG_PROD_OF_FRO_OVER_MARGIN = 36
    LOG_PROD_OF_FRO = 37
    LOG_SUM_OF_FRO_OVER_MARGIN = 38
    LOG_SUM_OF_FRO = 39
    FRO_DIST = 40
    DIST_SPEC_INIT = 41
    PARAM_NORM = 42
    PATH_NORM_OVER_MARGIN = 43
    PATH_NORM = 44
    PACBAYES_INIT = 48
    PACBAYES_ORIG = 49
    PACBAYES_FLATNESS = 53
    PACBAYES_MAG_INIT = 56
    PACBAYES_MAG_ORIG = 57
    PACBAYES_MAG_FLATNESS = 61
    # Other Measures
    L2 = 100
    L2_DIST = 101
    # FFT Spectral Measures
    LOG_SPEC_INIT_MAIN_FFT = 129
    LOG_SPEC_ORIG_MAIN_FFT = 130
    LOG_PROD_OF_SPEC_OVER_MARGIN_FFT = 131
    LOG_PROD_OF_SPEC_FFT = 132
    FRO_OVER_SPEC_FFT = 133
    LOG_SUM_OF_SPEC_OVER_MARGIN_FFT = 134
    LOG_SUM_OF_SPEC_FFT = 135
    DIST_SPEC_INIT_FFT = 141


class OptimizerType(Enum):
    SGD = 1
    SGD_MOMENTUM = 2
    ADAM = 3
    NERO = 4


class LossType(Enum):
    CE = 1
    MSE = 2


class Verbosity(IntEnum):
    NONE = 1
    RUN = 2
    EPOCH = 3
    BATCH = 4


@dataclass(frozen=False)
class State:
    epoch: int = 1
    batch: int = 1
    global_batch: int = 1
    converged: bool = False
    check_freq: int = 0
    ce_check_milestones: Optional[List[float]] = None
    acc_check_milestones: Optional[List[float]] = None
    prior_weights: Optional[list] = None


# Hyperparameters that uniquely determine the experiment


@dataclass(frozen=True)
class HParams:
    seed: int = 0
    use_cuda: bool = True
    # Model
    model_type: ModelType = ModelType.FCN

    # for FCN and gui-CNN
    model_width_tuple: List[int] = field(
        default_factory=lambda: [1024, 1024])

    SI_w_std: Optional[float] = None # Only for scale-ignorant FCN initialization

    # for gui-CNN only
    intermediate_pooling_type: Optional[str] = None # can be "avg", "max"
    pooling: Optional[str] = "max" # can be "avg", "max"

    # for NiN
    base_width: Optional[int] = None # for NiN defalut is 25
    model_depth: Optional[int] = None # NiN default 2
    model_width: Optional[int] = None # NiN default 8

    # Dataset
    center_data: bool = False
    dataset_type: DatasetType = DatasetType.MNIST_binary
    data_seed: Optional[int] = 42
    train_dataset_size: Optional[int] = 2000
    test_dataset_size: Optional[int] = 10000
    label_corruption: Optional[float] = None # between [0, 1], the portion of training set that has random labels
    attack_dataset_size: Optional[int] = None
    # Training
    loss: LossType = LossType.CE
    batch_size: int = 256
    epochs: int = 300
    optimizer_type: OptimizerType = OptimizerType.ADAM
    lr: float = 0.01

    # Stopping criterion
    stop_by_full_train_acc: bool = True # Stop training when reaching 100% training accuracy
    # If this is set to be true, then ce_target* would be neglected.

    # Accuracy stopping criterion
    acc_target: float = 1.0
    acc_target_milestones: Optional[List[float]] = field(
            default_factory=lambda: [0.9, 0.95, 0.99])

    # Cross-entropy stopping criterion
    ce_target: Optional[float] = 0.01
    ce_target_milestones: Optional[List[float]] = field(
        default_factory=lambda: [0.05, 0.025, 0.015])
        # these two would be neglected if stop_by_full_train_acc=True

    # GP measures related
    compute_prior: bool = False
    compute_mar_lik: bool = True
    normalize_kernel: bool = False
    PU_MC: bool = False # Use MC method to calculate PU, as proposed in Jeremy v1
    PU_EP: bool = True
    PU_MC_sample: int = 100000
    use_empirical_K:bool = False
    delta: Optional[float] = None # the damping factor 
                                  # ("learning rate") of EP site-parameter update.
                                  # if None then 1.0

    optimize_PAC_Bayes_bound: bool = False

    def to_tensorboard_dict(self) -> dict:
        d = asdict(self)
        d = {x: y for (x, y) in d.items() if y is not None}
        d = {x: (y.name if isinstance(y, Enum) else y) for x, y in d.items()}
        return d

    @property
    def md5(self):
        return hashlib.md5(str(self).encode('utf-8')).hexdigest()

    @property
    def wandb_md5(self):
        dictionary = self.to_tensorboard_dict()
        dictionary['seed'] = 0
        dictionary['data_seed'] = 0
        dictionary['use_cuda'] = False
        dictionary['delta'] = None
        # This puts experiments under different initializations into the same wandb group
        return hashlib.md5(str(dictionary).encode('utf-8')).hexdigest()

# Configuration which doesn't affect experiment results


@dataclass(frozen=True)
class Config:
    id: int = field(default_factory=lambda: time.time())
    log_batch_freq: Optional[int] = None
    log_epoch_freq: Optional[int] = 10
    save_epoch_freq: Optional[int] = 1
    root_dir: Path = Path('./temp')
    data_dir: Path = Path('./temp/data')
    verbosity: Verbosity = Verbosity.EPOCH
    use_tqdm: bool = False

    def setup_dirs(self) -> None:
        # Set up directories
        for directory in ('results', 'checkpoints'):
            (self.root_dir / directory).mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_dir(self):
        return self.root_dir / 'checkpoints'

    @property
    def results_dir(self):
        return self.root_dir / 'results'


class EvaluationMetrics(NamedTuple):
    acc: float
    avg_loss: float
    num_correct: int
    num_to_evaluate_on: int
    all_complexities: Dict[ComplexityType, float]
