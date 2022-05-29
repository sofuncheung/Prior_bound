from copy import deepcopy
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from .dataset_helpers import get_dataloaders
from .experiment_config import (
    Config,
    ModelType,
    DatasetSubsetType,
    HParams,
    State,
    EvaluationMetrics,
    OptimizerType,
)
from .logs import BaseLogger, Printer
from .measures import get_all_measures
from .models import *


class Experiment:
    def __init__(
        self,
        state: State,
        device: torch.device,
        hparams: HParams,
        config: Config,
        logger: BaseLogger,
        result_save_callback: Optional[object] = None
    ):
        self.state = state
        self.device = device
        self.hparams = hparams
        self.config = config

        # Random Seeds
        random.seed(self.hparams.seed)
        np.random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)
        torch.cuda.manual_seed_all(self.hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Logging
        self.logger = logger
        # Printing
        self.printer = Printer(self.config.id, self.config.verbosity)
        self.result_save_callback = result_save_callback

        # Model
        self.model = self._get_model(self.hparams)
        print("Number of parameters", sum(p.numel()
              for p in self.model.parameters() if p.requires_grad))
        self.model.to(device)
        self.init_model = deepcopy(self.model)
        if (self.hparams.model_type == ModelType.FCN or
                self.hparams.model_type) == ModelType.NiN:
            self.model_fc_popped = None
        else:
            self.model_fc_popped = self._get_model_fc_popped(self.hparams)

        # Optimizer
        if self.hparams.optimizer_type == OptimizerType.SGD:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer_type == OptimizerType.SGD_MOMENTUM:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.hparams.lr, momentum=0.9)
        elif self.hparams.optimizer_type == OptimizerType.ADAM:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.hparams.lr)
        else:
            raise KeyError

        # Check and load if the experiment was run and saved before.
        self.checkNload()

        # Load data
        (self.train_loader, self.train_eval_loader,
                self.test_loader) = get_dataloaders(
                        self.hparams, self.config, self.device)

    @staticmethod
    def _get_model(hparams: HParams) -> ModelType:
        if hparams.model_type == ModelType.NiN:
            return NiN(hparams.model_depth, hparams.model_width,
                    hparams.base_width, hparams.dataset_type)
        elif hparams.model_type == ModelType.FCN:
            return FCN(hparams.model_width_tuple, hparams.dataset_type)
        elif hparams.model_type == ModelType.CNN:
            return CNN(hparams.model_width_tuple,
                    hparams.intermediate_pooling_type,
                    hparams.pooling,
                    hparams.dataset_type)

    @staticmethod
    def _get_model_fc_popped(hparams: HParams) -> ModelType:
        if hparams.model_type == ModelType.CNN:
            return CNN_fc_popped(hparams.model_width_tuple,
                    hparams.intermediate_pooling_type,
                    hparams.pooling,
                    hparams.dataset_type)


    def save_state(self, postfix: str = '') -> None:
        checkpoint_file = self.config.checkpoint_dir / \
            (self.hparams.md5 + postfix + '.pt')
        torch.save({
            'config': self.hparams,
            'state': self.state,
            'model': self.model.state_dict(),
            'init_model': self.init_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'np_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
        }, checkpoint_file)

    def checkNload(self,) -> None:
        """
        Check whether there is corresponding pt file,
        if so, check if it's converged.
        If not, load it.
        """
        checkpoint_file = self.config.checkpoint_dir / \
            (self.hparams.md5 + '.pt')
        try:
            ckpt = torch.load(checkpoint_file)
            if ckpt['state'].converged == True:
                print("Checkpoint file found but already converged. Fresh start.")
            else:
                print("Checkpoint file found, loading model...")
                self.model.load_state_dict(ckpt['model'])
                print("Loading init_model")
                self.init_model.load_state_dict(ckpt['init_model'])
                print("Loading optimizer")
                self.optimizer.load_state_dict(ckpt['optimizer'])

        except FileNotFoundError:
            print("No checkpoint file found. Fresh start.")


    def _train_epoch(self) -> None:
        self.model.train()
        check_batches = [(len(self.train_loader)//(2**(self.state.check_freq))) * (i+1)
                            for i in range(2**(self.state.check_freq)-1)]
        check_batches.append(len(self.train_loader)-1) # mandatorily check last batch
        if 2**(self.state.check_freq) > len(self.train_loader):
            check_batches = [i for i in range(len(self.train_loader))]
            # when state.check_freq is high enough, check every batch
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device, non_blocking=True), target.to(
                self.device, non_blocking=True)

            self.state.batch = batch_idx
            self.state.global_batch = (
                self.state.epoch - 1) * len(self.train_loader) + self.state.batch
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model(data).squeeze(-1)
            cross_entropy = F.binary_cross_entropy_with_logits(logits, target)

            cross_entropy.backward()
            loss = cross_entropy.clone()

            self.model.train()

            self.optimizer.step()

            # Log everything
            self.printer.batch_end(
                self.config, self.state, data, self.train_loader, loss)
            self.logger.log_batch_end(
                self.config, self.state, cross_entropy, loss)

            if batch_idx == check_batches[0]:
                check_batches.pop(0)
                is_last_batch = batch_idx == (len(self.train_loader)-1)
                if self.hparams.stop_by_full_train_acc == True:
                    # Accuracy stopping check
                    train_acc = self.evaluate_cross_entropy(
                            DatasetSubsetType.TRAIN, log=is_last_batch)[1]
                    if train_acc == self.hparams.acc_target:
                        self.state.converged = True
                    else:
                        while (len(self.state.acc_check_milestones) > 0
                        and train_acc >= self.state.acc_check_milestones[0]):
                            passed_milestone = self.state.acc_check_milestones[0]
                            print(f'passed acc milestone {passed_milestone}')
                            self.state.acc_check_milestones.pop(0)
                            self.state.check_freq += 1
                            # if passed one milestone, double the check frequency
                            if self.config.save_epoch_freq is not None:
                                self.save_state(f'_acc_{passed_milestone}')

                else:
                    # Cross-entropy stopping check
                    dataset_ce = self.evaluate_cross_entropy(
                        DatasetSubsetType.TRAIN, log=is_last_batch)[0]
                    if dataset_ce < self.hparams.ce_target:
                        self.state.converged = True
                    else:
                        while len(self.state.ce_check_milestones) > 0 and dataset_ce <= self.state.ce_check_milestones[0]:
                            passed_milestone = self.state.ce_check_milestones[0]
                            print(f'passed ce milestone {passed_milestone}')
                            self.state.ce_check_milestones.pop(0)
                            self.state.check_freq += 1
                            # if passed one milestone, double the ce check frequency
                            if self.config.save_epoch_freq is not None:
                                self.save_state(f'_ce_{passed_milestone}')

            if self.state.converged:
                break

    def train(self) -> None:
        self.printer.train_start(self.device)
        train_eval, val_eval = None, None

        self.state.global_batch = 0
        for epoch in trange(self.state.epoch, self.hparams.epochs + 1, disable=(not self.config.use_tqdm)):
            self.state.epoch = epoch
            self._train_epoch()

            is_evaluation_epoch = (
                epoch == 1 or epoch == self.hparams.epochs or epoch % self.config.log_epoch_freq == 0)
            if is_evaluation_epoch or self.state.converged:
                train_eval = self.evaluate(
                    DatasetSubsetType.TRAIN, (epoch == self.hparams.epochs or self.state.converged))
                val_eval = self.evaluate(DatasetSubsetType.TEST)
                self.logger.log_generalization_gap(
                    self.state, train_eval.acc, val_eval.acc, train_eval.avg_loss, val_eval.avg_loss, train_eval.all_complexities)
                self.printer.epoch_metrics(epoch, train_eval, val_eval)

            if epoch == self.hparams.epochs or self.state.converged:
                self.result_save_callback(epoch, val_eval, train_eval)

            # Save state
            is_save_epoch = self.config.save_epoch_freq is not None and (
                epoch % self.config.save_epoch_freq == 0 or epoch == self.hparams.epochs or self.state.converged)
            if is_save_epoch:
                self.save_state()

            if self.state.converged:
                print('Converged')
                break

        self.printer.train_end()

        if train_eval is None or val_eval is None:
            raise RuntimeError

    @torch.no_grad()
    def evaluate(self, dataset_subset_type: DatasetSubsetType, compute_all_measures: bool = False) -> EvaluationMetrics:
        self.model.eval()
        data_loader = [self.train_eval_loader,
                       self.test_loader][dataset_subset_type]
        trainNtest_loaders = (self.train_eval_loader, self.test_loader)

        cross_entropy_loss, acc, num_correct = self.evaluate_cross_entropy(
            dataset_subset_type)

        all_complexities = {}
        if dataset_subset_type == DatasetSubsetType.TRAIN and compute_all_measures:
            all_complexities = get_all_measures(
                self.model, self.init_model, self.model_fc_popped,
                trainNtest_loaders, acc, self.hparams.seed, self.hparams.model_type,
                self.hparams.compute_mar_lik, self.hparams.compute_prior,
                self.hparams.normalize_kernel)

        self.logger.log_epoch_end(
            self.hparams, self.state, dataset_subset_type, cross_entropy_loss, acc)

        return EvaluationMetrics(acc, cross_entropy_loss, num_correct, len(data_loader.dataset), all_complexities)

    @torch.no_grad()
    def evaluate_cross_entropy(self, dataset_subset_type: DatasetSubsetType, log: bool = False):
        self.model.eval()
        cross_entropy_loss = 0
        num_correct = 0

        data_loader = [self.train_eval_loader,
                       self.test_loader][dataset_subset_type]
        num_to_evaluate_on = len(data_loader.dataset)

        for data, target in data_loader:
            data, target = data.to(self.device, non_blocking=True), target.to(
                self.device, non_blocking=True)
            logits = self.model(data).squeeze(-1)
            cross_entropy = F.binary_cross_entropy_with_logits(logits, target, reduction='sum')
            cross_entropy_loss += cross_entropy.item()  # sum up batch loss

            # get the index of the max logits
            pred = logits.data > 0
            batch_correct = pred.eq(target.data.view_as(
                pred)).type(torch.FloatTensor).cpu()
            num_correct += batch_correct.sum()

        cross_entropy_loss /= num_to_evaluate_on
        acc = num_correct.item() / num_to_evaluate_on

        if log:
            self.logger.log_epoch_end(
                self.hparams, self.state, dataset_subset_type, cross_entropy_loss, acc)
        return cross_entropy_loss, acc, num_correct
