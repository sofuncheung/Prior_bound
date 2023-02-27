import sys
from contextlib import contextmanager
from copy import deepcopy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from .experiment_config import ComplexityType as CT
from .experiment_config import ModelType, LossType, HParams
from .models import ExperimentBaseModel
from .empirical_kernel import empirical_K
from .fc_kernel import *
from .GP_prob.GP_prob_gpy2 import GP_prob, nngp_Heaviside_likelihood_posterior
from .GP_prob.GP_prob_regression import  GP_prob_regression_KL, GP_prob_pf
from .GP_prob.GP_prob_regression import GP_regression_noiseless_posterior
from .GP_prob.nngp_mse_heaviside_posterior import nngp_mse_heaviside_posteror_params

from .util.GauOrthProb import MC_complexity_N_ELBO
from .util.ScdPACBayesBound import AveGibbsAgreement

# Adapted from https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py

# This function reparametrizes the networks with batch normalization in a way that
# it calculates the same function as the original network but without batch normalization
# Instead of removing batch norm completely, we set the bias and mean
# to zero, and scaling and variance to one
# Warning: This function only works for convolutional and fully connected networks.
# It also assumes that module.children() returns the children of a module
# in the forward pass order. Recurssive construction is allowed.

# (24 Feb 2023) Note also this wouldn't work with Conv or FC layer without biases.
# So I decided to turn on biases terms in ResNet.

@torch.no_grad()
def _reparam(model):
    def in_place_reparam(model, prev_layer=None):
        for child in model.children():
            prev_layer = in_place_reparam(child, prev_layer)
            if child._get_name() == 'Conv2d':
                prev_layer = child
                # prev_layer is the layer BEFORE batch_norm layer
                # The strategy here is to reparameterise this prev_layer
                # to offset the batchnorm layer behind it.
            elif child._get_name() == 'BatchNorm2d':
                # gamma, beta, running_mean, running_var are all
                # [n_features] vectors.
                # print("batchnorm.weight (gamma):", child.weight.shape)
                # print("batchnorm.bias (beta):", child.bias.shape)
                # print("running mean:", child.running_mean.shape)
                # print("running var:", child.running_var.shape)
                scale = child.weight / ((child.running_var + child.eps).sqrt())
                prev_layer.bias.copy_(
                    child.bias + (scale * (prev_layer.bias - child.running_mean)))
                perm = list(reversed(range(prev_layer.weight.dim())))
                prev_layer.weight.copy_(
                    (prev_layer.weight.permute(perm) * scale).permute(perm))
                child.bias.fill_(0)
                child.weight.fill_(1)
                child.running_mean.fill_(0)
                child.running_var.fill_(1)
        return prev_layer
    model = deepcopy(model)
    in_place_reparam(model)
    return model


@torch.no_grad()
def _reparam_densenet(model):
    r"""
    DenseNet is pain in the ass: the batchnorm layer is sometimes before and
    sometimes after the conv (or the final FC) layers. So it requires different
    treatment.

    Actully, after more carefully thinking about it, you just can not reparam conv
    layer to make up for the batchnorm in Densenet. The batchnorm before conv has a
    ReLU between them so the transformation is non-linear. The batchnorm after conv
    layers would be in another _Denselayer so the num_features will be different
    because of the feature concatenation nature of DenseNets.

    So, this whole function is bullshit. I'm keeping it to embody the
    inevitable daunting process of developing.
    """
    def module_is_minimal_layer(m):
        """
        Check if a module is the basic layer or a module that contains more than one
        layers.
        """
        if len(list(m.children())) == 0:
            return True
        else:
            return False

    def module_is_subminimal(m):
        """
        Check if a module only contains minimal layers
        """
        return (all([module_is_minimal_layer(l) for l in list(m.children())])
                and not module_is_minimal_layer(m))

    def get_full_name(modules_dict, layer):
        key_list = list(modules_dict.keys())
        val_list = list(modules_dict.values())
        position = val_list.index(layer)
        return(key_list[position])

    def corr_bn(modules_dict, layer):
        """
        Return the batchnorm layer that is corresponding to the given
        Conv2d or Linear layer in a DenseNet model.
        """
        full_name = get_full_name(modules_dict, layer)
        #if child._get_name() == "Conv2d":
        #    bn = 

        return(get_full_name(modules_dict, layer))

    def in_place_reparam(all_named_modules, model):
        """
        Minimal module refers to a module that only contains minimal layers.
        Examples are (denselayer5), (trainsition1), etc.
        """
        for child in model.children():
            in_place_reparam(all_named_modules, child)
            #print(get_full_name(all_named_modules, child))
            if child._get_name() == "_DenseLayer":
                for conv_name in ["conv1", "conv2"]:
                    conv_layer = child.get_submodule(conv_name)
                    bn_layer = child.get_submodule(conv_name.replace("conv","norm"))
                    scale = bn_layer.weight / (
                            (bn_layer.running_var + bn_layer.eps).sqrt())
                    perm = list(reversed(range(conv_layer.weight.dim())))
                    conv_layer.bias.copy_(
                            conv_layer.bias
                            + conv_layer.weight.permute(perm) * (
                                bn_layer.bias - scale * bn_layer.running_mean))
                    conv_layer.weight.copy_(
                            (conv_layer.weight.permute(perm) * scale).permute(perm))
            #    print(corr_bn(all_named_modules, child))
            #print(child._get_name())
            #if child._get_name() in ["Conv2d", "Linear"]:
                # Finding corresponding batchnorm layer
            #    print(corr_bn(all_named_modules, child))
    model = deepcopy(model)
    all_named_modules = dict(model.named_modules())
    in_place_reparam(all_named_modules, model)
    sys.exit()
    return model


@contextmanager
# The use of contextmanager see: https://docs.python.org/3/library/contextlib.html
def _perturbed_model(
    model: ExperimentBaseModel,
    sigma: float,
    rng,
    magnitude_eps: Optional[float] = None
):
    device = next(model.parameters()).device # Here next is just to get an item from the
                                             # generator.
    if magnitude_eps is not None:
        noise = [torch.normal(0, sigma**2 * torch.abs(p) ** 2 +
                    magnitude_eps ** 2, generator=rng) for p in model.parameters()]
    else:
        noise = [torch.normal(0, sigma**2, p.shape, generator=rng).to(device)
                 for p in model.parameters()]
    model = deepcopy(model)
    try:
        [p.add_(n) for p, n in zip(model.parameters(), noise)]
        yield model
    finally:
        [p.sub_(n) for p, n in zip(model.parameters(), noise)]
        del model


def FCN_with_noise(model, log_post_std_list, rng, x):
    """
    Alert!!! Only works with FCN defined in .models
    For PAC-Bayes bound optimization
    Re-write the forward rule by hand, featuring the fact that the
    weights of layers have two part: orginal weights and (noise x std)
    and they are both learnable.
    """
    w_b_list = list(model.parameters())
    device = w_b_list[0].device
    simple_noise = [torch.normal(0, 1, p.shape, generator=rng).to(device)
                    for p in w_b_list]

    num_hidden_layers = len(w_b_list)/2 - 1
    # flattening data
    x = torch.flatten(x, start_dim=1)
    for idx, (w, b, s_w, s_b, n_w, n_b) in enumerate(zip(
            w_b_list[::2], w_b_list[1::2],
            log_post_std_list[::2], log_post_std_list[1::2],
            simple_noise[::2], simple_noise[1::2])):
        x = torch.matmul(x, (w + torch.mul(torch.exp(s_w), n_w)).T) + b + torch.mul(
                torch.exp(s_b), n_b)
        if idx < num_hidden_layers:
            x = F.relu(x)

    return x


# Adapted from https://drive.google.com/file/d/1_6oUG94d0C3x7x2Vd935a2QqY-OaAWAM/view
def _pacbayes_sigma(
    model: ExperimentBaseModel,
    dataloader: DataLoader,
    accuracy: float,
    seed: int,
    magnitude_eps: Optional[float] = None,
    search_depth: int = 15,
    montecarlo_samples: int = 10,
    accuracy_displacement: float = 0.1,
    displacement_tolerance: float = 1e-2,
) -> float:
    lower, upper = 0, 2
    sigma = 1

    BIG_NUMBER = 10348628753
    device = next(model.parameters()).device
    rng = torch.Generator(
        device=device) if magnitude_eps is not None else torch.Generator()
    rng.manual_seed(BIG_NUMBER + seed)

    for _ in range(search_depth):
        sigma = (lower + upper) / 2
        accuracy_samples = []
        for _ in range(montecarlo_samples):
            with _perturbed_model(model, sigma, rng, magnitude_eps) as p_model:
                loss_estimate = 0
                for data, target in dataloader:
                    logits = p_model(data)
                    pred = logits.data > 0
                    batch_correct = pred.eq(target.data.view_as(
                        pred)).type(torch.FloatTensor).cpu()
                    loss_estimate += batch_correct.sum()
                loss_estimate /= len(dataloader.dataset)
                accuracy_samples.append(loss_estimate)
        displacement = abs(np.mean(accuracy_samples) - accuracy)
        if abs(displacement - accuracy_displacement) < displacement_tolerance:
            break
        elif displacement > accuracy_displacement:
            # Too much perturbation
            upper = sigma
        else:
            # Not perturbed enough to reach target displacement
            lower = sigma
    return sigma


@torch.no_grad()
def get_all_measures(
    model: ExperimentBaseModel,
    init_model: ExperimentBaseModel,
    model_fc_popped: ExperimentBaseModel,
    trainNtest_loaders: Tuple[DataLoader, DataLoader],
    acc: float,
    hparams: HParams
) -> Dict[CT, float]:
    seed = hparams.seed
    model_type = hparams.model_type
    compute_mar_lik = hparams.compute_mar_lik
    compute_prior = hparams.compute_prior
    optimize_PAC_Bayes_bound = hparams.optimize_PAC_Bayes_bound
    normalize_kernel = hparams.normalize_kernel
    PU_EP = hparams.PU_EP
    PU_MC = hparams.PU_MC
    PU_MC_sample = hparams.PU_MC_sample
    loss = hparams.loss
    use_empirical_K = hparams.use_empirical_K

    measures = {}

    if model_type in [ModelType.FCN, ModelType.CNN, ModelType.RESNET50,
            ModelType.NiN, ModelType.FCN_SI]:
        model = _reparam(model)
        init_model = _reparam(init_model)
    elif model_type in [ModelType.DENSENET121]:
        pass

    device = next(model.parameters()).device
    device_string = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

    m = len(trainNtest_loaders[0].dataset)

    def _get_xs_ys_from_dataset(dataset):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=500,
            shuffle=False,
            num_workers=0,
            drop_last=False
            )
        xs = []
        ys = []
        for batch_idx, (inputs, targets) in enumerate(loader):
            xs.append(inputs.reshape(inputs.shape[0],-1))
            ys.append(targets)
        xs = torch.cat(xs, axis=0)
        ys = torch.cat(ys, axis=0)
        return (xs,ys)

    def _get_label_prediction(model, dataset, loss):
        device = next(model.parameters()).device
        model.eval()
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=500,
            shuffle=False,
            num_workers=0)
        outputs_list = []
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            with torch.no_grad():
                logits = model(inputs)
                if loss == LossType.CE:
                    pred = logits.data > 0
                    # funny thing, I made a mistake here which makes all
                    # pred to be zero. In this case the prior bound would be
                    # very small. That means this simple function is very likely
                    # to be found.
                elif loss == LossType.MSE:
                    #pred = torch.where(logits.data>0, 1, -1)
                    pred = logits.data # Using logits to calculate PU in regression.
                outputs_list.append(pred)

        return torch.cat(outputs_list, axis=0)

    if compute_prior:
        print("GP based measures")
        data_train_plus_test = torch.utils.data.ConcatDataset(
                (trainNtest_loaders[0].dataset,
                trainNtest_loaders[1].dataset))

        (xs, _ys_labels) = _get_xs_ys_from_dataset(data_train_plus_test)
        ys = _get_label_prediction(model, data_train_plus_test, loss)

        if xs.is_cuda:
            xs = xs.cpu()
        if ys.is_cuda:
            ys = ys.cpu()
        xs_train = xs[:m]
        xs_test = xs[m:]

        if model_type in [ModelType.FCN, ModelType.FCN_SI] and use_empirical_K == False:
            # Use analytical arccos kernel
            K = kernel_matrix(xs, model.number_layers, np.sqrt(2), 0.1)
        elif use_empirical_K == True:
            K = empirical_K(model_fc_popped, data_train_plus_test,
                    #2,
                    0.1*len(data_train_plus_test),
                    np.sqrt(2),
                    0.1,
                    device_string, seed,
                    n_gpus=1,
                    empirical_kernel_batch_size=500,
                    truncated_init_dist=False,
                    store_partial_kernel=False,
                    partial_kernel_n_proc=1,
                    partial_kernel_index=0
                    )
        else:
            raise NotImplementedError("GP kernel not calculated!")

            K = np.array(K.cpu())
            # gpy EP calculation uses np.float64 internally. So it doesn't matter what precision
            # you use here.
        if normalize_kernel:
            K = K / K.max()
        if loss == LossType.CE:
            if PU_EP == True: # Use EP approximation to calculate PU
                logPU = GP_prob(K, np.array(xs), np.array(ys))
                measures[CT.PRIOR] = torch.tensor((-logPU-np.log(2**-10))/m, device=device, dtype=torch.float32)
            if PU_MC == True: # Use MC method in Jeremy v1 to approximate PU
                ys = 2 * ys - 1 # here ys need to be 1 and -1.
                c0, c1 = MC_complexity_N_ELBO(K, np.array(ys.float()), PU_MC_sample)
                measures[CT.PRIOR_MC] = torch.tensor((c0-np.log(2**-10))/m, device=device, dtype=torch.float32)
                measures[CT.PRIOR_ELBO] = torch.tensor((c1-np.log(2**-10))/m, device=device, dtype=torch.float32)
            #'''
            ys_train = _ys_labels[:m]
            if ys_train.is_cuda:
                ys_train = ys_train.cpu()
            ys_train = [[y] for y in ys_train]
            posterior_GPy = nngp_Heaviside_likelihood_posterior(
                    np.array(xs_train), np.array(ys_train), np.array(xs_test), K)
            print(posterior_GPy)
            print(AveGibbsAgreement(posterior_GPy))
            #'''
        elif loss == LossType.MSE:
            # Compute the probabilities (prior and mar_lik etc) for GP regression.
            logPU = GP_prob_pf(K, np.array(xs), np.array(ys))
            measures[CT.PRIOR] = torch.tensor((-logPU-np.log(2**-10))/m, device=device, dtype=torch.float32)

            ys_train = ys[:m]
            posterior_GPy = nngp_mse_heaviside_posteror_params(np.array(xs_train), np.array(ys_train), np.array(xs_test), K, noise=False)
            # print(posterior_GPy)

        # Marginal likelihood
        if compute_mar_lik:
            K_marg = K[:m, :m]
            K_cross = K[:m, m:]
            k_diag_test = K[m:, m:].diagonal()
            (xs_train, ys_train) = _get_xs_ys_from_dataset(trainNtest_loaders[0].dataset)
            if loss == LossType.MSE:
                ys_train = ys[:m]
            if xs_train.is_cuda:
                xs_train = xs_train.cpu()
            if ys_train.is_cuda:
                ys_train = ys_train.cpu()
            ys_train = [[y] for y in ys_train]

            if loss == LossType.CE:
                if PU_EP == True: # Use EP approximation to calculate PU
                    logPS = GP_prob(K_marg, np.array(xs_train), np.array(ys_train))
                    mar_lik_bound = (-logPS + 2*np.log(m) + 1 - np.log(2**-10)) / m
                    # "Some PAC-Bayesian Theorems" by McAllester 1999, Theorem 1.
                    mar_lik_bound = 1-np.exp(-mar_lik_bound)

                    measures[CT.MAR_LIK] = torch.tensor(mar_lik_bound, device=device, dtype=torch.float32)
                if PU_MC == True: # Use MC method in Jeremy v1 to approximate PU
                    ys_train = 2. * np.array(ys_train) - 1
                    c0, c1 = MC_complexity_N_ELBO(K_marg, ys_train, PU_MC_sample)
                    mar_lik_mc = (c0 + 2*np.log(m) + 1 - np.log(2**-10)) / m
                    mar_lik_elbo = (c1 + 2*np.log(m) + 1 - np.log(2**-10)) / m
                    mar_lik_mc = 1-np.exp(-mar_lik_mc)
                    mar_lik_elbo = 1-np.exp(-mar_lik_elbo)
                    measures[CT.MAR_LIK_MC] = torch.tensor(mar_lik_mc, device=device, dtype=torch.float32)
                    measures[CT.MAR_LIK_ELBO] = torch.tensor(mar_lik_elbo, device=device, dtype=torch.float32)
            elif loss == LossType.MSE:
                logPS = GP_prob_pf(K_marg, np.array(xs_train), np.array(ys_train))
                mar_lik_bound = (-logPS + 2*np.log(m) + 1 - np.log(2**-10)) / m
                mar_lik_bound = 1-np.exp(-mar_lik_bound)
                measures[CT.MAR_LIK] = torch.tensor(mar_lik_bound, device=device, dtype=torch.float32)
                noiseless_posterior = GP_regression_noiseless_posterior(K_marg, K_cross, k_diag_test, np.array(ys_train))
                # print(noiseless_posterior[0], noiseless_posterior[1])
                # print((posterior_GPy[0].squeeze(1)-noiseless_posterior[0]).max())
                # print((posterior_GPy[1].squeeze(1)-noiseless_posterior[1]).max())


    if compute_mar_lik and (not compute_prior):
        print("GP based measures")
        (xs_train, ys_train) = _get_xs_ys_from_dataset(trainNtest_loaders[0].dataset)
        (xs_test, _) = _get_xs_ys_from_dataset(trainNtest_loaders[1].dataset)
        if loss == LossType.MSE:
            ys_train = _get_label_prediction(model, trainNtest_loaders[0].dataset, loss)
        if xs_train.is_cuda:
            xs_train = xs_train.cpu()
        if xs_test.is_cuda:
            xs_test = xs_test.cpu()
        if ys_train.is_cuda:
            ys_train = ys_train.cpu()
        ys_train = [[y] for y in ys_train]

        if model_type in [ModelType.FCN, ModelType.FCN_SI] and use_empirical_K == False:
            K_marg = kernel_matrix(xs_train, model.number_layers, np.sqrt(2), 0.1)

            '''
            K_cross = kernel_matrix_cross(xs_train, xs_test, model.number_layers, np.sqrt(2), 0)
            k_diag_test = k_diag_vector(xs_test, model.number_layers, np.sqrt(2), 0)
            '''
        elif use_empirical_K == True:
            K_marg = empirical_K(model_fc_popped, trainNtest_loaders[0].dataset,
                #2,
                0.1*m,
                np.sqrt(2),
                0.1,
                device_string, seed,
                n_gpus=1,
                empirical_kernel_batch_size=500,
                truncated_init_dist=False,
                store_partial_kernel=False,
                partial_kernel_n_proc=1,
                partial_kernel_index=0
                )
            K_marg = np.array(K_marg.cpu())
        else:
            raise NotImplementedError("GP kernel not calculated!")

        if normalize_kernel:
            K_marg = K_marg / K_marg.max()

        if loss == LossType.CE:
            if PU_EP == True: # Use EP approximation to calculate PU
                logPS = GP_prob(K_marg, np.array(xs_train), np.array(ys_train))
                # 
                #np.save('xs.npy',np.array(xs_train))
                #np.save('ys.npy',np.array(ys_train))
                #sys.exit()
                mar_lik_bound = (-logPS + 2*np.log(m) + 1 - np.log(2**-10)) / m
                mar_lik_bound = 1-np.exp(-mar_lik_bound)

                measures[CT.MAR_LIK] = torch.tensor(mar_lik_bound, device=device, dtype=torch.float32)

            if PU_MC == True: # Use MC method in Jeremy v1 to approximate PU
                ys_train = 2. * np.array(ys_train) - 1
                c0, c1 = MC_complexity_N_ELBO(K_marg, ys_train, PU_MC_sample)
                mar_lik_mc = (c0 + 2*np.log(m) + 1 - np.log(2**-10)) / m
                mar_lik_elbo = (c1 + 2*np.log(m) + 1 - np.log(2**-10)) / m
                mar_lik_mc = 1-np.exp(-mar_lik_mc)
                mar_lik_elbo = 1-np.exp(-mar_lik_elbo)
                measures[CT.MAR_LIK_MC] = torch.tensor(mar_lik_mc, device=device, dtype=torch.float32)
                measures[CT.MAR_LIK_ELBO] = torch.tensor(mar_lik_elbo, device=device, dtype=torch.float32)
        elif loss == LossType.MSE:
            logPS = GP_prob_pf(K_marg, np.array(xs_train), np.array(ys_train))
            mar_lik_bound = (-logPS + 2*np.log(m) + 1 - np.log(2**-10)) / m
            mar_lik_bound = 1-np.exp(-mar_lik_bound)
            measures[CT.MAR_LIK] = torch.tensor(mar_lik_bound, device=device, dtype=torch.float32)
            noiseless_posterior = GP_regression_noiseless_posterior(K_marg, K_cross, k_diag_test, np.array(ys_train))
            print(AveGibbsAgreement(noiseless_posterior))


    def get_weights_only(model: ExperimentBaseModel) -> List[Tensor]:
        blacklist = {'bias', 'bn'}
        return [p for name, p in model.named_parameters() if all(x not in name for x in blacklist)]

    weights = get_weights_only(model)
    dist_init_weights = [p-q for p,
                         q in zip(weights, get_weights_only(init_model))]
    d = len(weights)

    def get_vec_params(weights: List[Tensor]) -> Tensor:
        return torch.cat([p.view(-1) for p in weights], dim=0)

    w_vec = get_vec_params(weights)
    dist_w_vec = get_vec_params(dist_init_weights)
    num_params = len(w_vec)

    def get_reshaped_weights(weights: List[Tensor]) -> List[Tensor]:
        # If the weight is a tensor (e.g. a 4D Conv2d weight), it will be reshaped to a 2D matrix
        return [p.view(p.shape[0], -1) for p in weights] # p.shape[0] = number of out_features

    reshaped_weights = get_reshaped_weights(weights)
    dist_reshaped_weights = get_reshaped_weights(dist_init_weights)

    print("Vector Norm Measures")
    measures[CT.L2] = w_vec.norm(p=2)
    measures[CT.L2_DIST] = dist_w_vec.norm(p=2)

    print("VC-Dimension Based Measures")
    measures[CT.PARAMS] = torch.tensor(num_params)  # 20

    print("Measures on the output of the network")

    def _margin(
        model: ExperimentBaseModel,
        dataloader: DataLoader
    ) -> Tensor:
        # Is margin defined on single-output-logit NNs?
        # I don't know. Here is how I'd do it.
        # margin = |f(x)| * sgn(pred is correct)
        # This is correct. But this implementation is ugly.
        # can be more elegant by using something like
        # target = 2 * target - 1
        def _is_preds_are_correct(logits, target):
            signs = torch.zeros(len(logits), device=target.device)
            for i in range(len(logits)):
                if ((target[i] == 1 and logits[i] > 0) or
                    (target[i] == 0 and logits[i] <= 0)):
                    signs[i] = 1.
                elif ((target[i] == 1 and logits[i] <= 0) or
                      (target[i] == 0 and logits[i] > 0)):
                    signs[i] = -1.
            return signs

        if loss == LossType.MSE:
            raise NotImplementedError(
                "For MSE loss margin is problematic (because data labels are of +1/-1)")
        margins = []
        for data, target in dataloader:
            logits = model(data).squeeze(-1)
            margin = logits.abs() * _is_preds_are_correct(logits, target)
            margins.append(margin)
        return torch.cat(margins).kthvalue(m // 10)[0]

    margin = _margin(model, trainNtest_loaders[0]).abs()
    measures[CT.INVERSE_MARGIN] = torch.tensor(
        1, device=device) / margin ** 2  # 22

    print("(Norm & Margin)-Based Measures")
    fro_norms = torch.cat([p.norm('fro').unsqueeze(0) **
                          2 for p in reshaped_weights])
    spec_norms = torch.cat(
        [p.svd().S.max().unsqueeze(0) ** 2 for p in reshaped_weights]) # Largest singular value
    dist_fro_norms = torch.cat(
        [p.norm('fro').unsqueeze(0) ** 2 for p in dist_reshaped_weights])
    dist_spec_norms = torch.cat(
        [p.svd().S.max().unsqueeze(0) ** 2 for p in dist_reshaped_weights])

    print("Approximate Spectral Norm for CNN; Exact Spectral Norm for FCN")
    # Note that these use an approximation from [Yoshida and Miyato, 2017]
    # https://arxiv.org/abs/1705.10941 (Section 3.2, Convolutions)
    measures[CT.LOG_PROD_OF_SPEC] = spec_norms.log().sum()  # 32
    measures[CT.LOG_PROD_OF_SPEC_OVER_MARGIN] = measures[CT.LOG_PROD_OF_SPEC] - \
        2 * margin.log()  # 31
    measures[CT.LOG_SPEC_INIT_MAIN] = measures[CT.LOG_PROD_OF_SPEC_OVER_MARGIN] + \
        (dist_fro_norms / spec_norms).sum().log()  # 29
    measures[CT.FRO_OVER_SPEC] = (fro_norms / spec_norms).sum()  # 33
    measures[CT.LOG_SPEC_ORIG_MAIN] = measures[CT.LOG_PROD_OF_SPEC_OVER_MARGIN] + \
        measures[CT.FRO_OVER_SPEC].log()  # 30
    measures[CT.LOG_SUM_OF_SPEC_OVER_MARGIN] = math.log(
        d) + (1/d) * (measures[CT.LOG_PROD_OF_SPEC] - 2 * margin.log())  # 34
    measures[CT.LOG_SUM_OF_SPEC] = math.log(
        d) + (1/d) * measures[CT.LOG_PROD_OF_SPEC]  # 35
    if model_type in []:
        print("Exact Spectral Norm for CNN Kernel Weights, Fully Connected layer weights not included")
        # Proposed in https://arxiv.org/abs/1805.10408
        # Adapted from https://github.com/brain-research/conv-sv/blob/master/conv2d_singular_values.py#L52

        # Note on 24 Apr 2022:
        # Right now I can't make sure whether CNN spectral norm bound should include the weights of
        # last fully connected layer, because it's 3:40 AM and I'm bloody sleepy.
        # Answers are likely to be found in https://arxiv.org/abs/1801.00171

        def _spectral_norm_fft(kernel: Tensor, input_shape: Tuple[int, int]) -> Tensor:
            # PyTorch conv2d filters use Shape(out,in,kh,kw)
            # [Sedghi 2018] code expects filters of Shape(kh,kw,in,out)
            # Pytorch doesn't support complex FFT and SVD, so we do this in numpy
            np_kernel = np.einsum('oihw->hwio', kernel.data.cpu().numpy())
            transforms = np.fft.fft2(np_kernel, input_shape, axes=[
                                     0, 1])  # Shape(ih,iw,in,out)
            singular_values = np.linalg.svd(
                transforms, compute_uv=False)  # Shape(ih,iw,min(in,out))
            spec_norm = singular_values.max()
            return torch.tensor(spec_norm, device=kernel.device)

        input_shape = (model.dataset_type.D[1], model.dataset_type.D[2])
        fft_spec_norms = torch.cat(
            [_spectral_norm_fft(p, input_shape).unsqueeze(0) ** 2
                for p in weights if len(p.shape)==4])
        fft_dist_spec_norms = torch.cat([_spectral_norm_fft(
            p, input_shape).unsqueeze(0) ** 2 for p in dist_init_weights
            if len(p.shape)==4])

        measures[CT.LOG_PROD_OF_SPEC_FFT] = fft_spec_norms.log().sum()  # 32
        measures[CT.LOG_PROD_OF_SPEC_OVER_MARGIN_FFT] = measures[CT.LOG_PROD_OF_SPEC_FFT] - \
            2 * margin.log()  # 31
        measures[CT.FRO_OVER_SPEC_FFT] = (fro_norms / fft_spec_norms).sum()  # 33
        measures[CT.LOG_SUM_OF_SPEC_OVER_MARGIN_FFT] = math.log(
            d) + (1/d) * (measures[CT.LOG_PROD_OF_SPEC_FFT] - 2 * margin.log())  # 34
        measures[CT.LOG_SUM_OF_SPEC_FFT] = math.log(
            d) + (1/d) * measures[CT.LOG_PROD_OF_SPEC_FFT]  # 35
        measures[CT.DIST_SPEC_INIT_FFT] = fft_dist_spec_norms.sum()  # 41
        measures[CT.LOG_SPEC_INIT_MAIN_FFT] = measures[CT.LOG_PROD_OF_SPEC_OVER_MARGIN_FFT] + \
            (dist_fro_norms / fft_spec_norms).sum().log()  # 29
        measures[CT.LOG_SPEC_ORIG_MAIN_FFT] = measures[CT.LOG_PROD_OF_SPEC_OVER_MARGIN_FFT] + \
            measures[CT.FRO_OVER_SPEC_FFT].log()  # 30

    print("Frobenius Norm")
    measures[CT.LOG_PROD_OF_FRO] = fro_norms.log().sum()  # 37
    measures[CT.LOG_PROD_OF_FRO_OVER_MARGIN] = measures[CT.LOG_PROD_OF_FRO] - \
        2 * margin.log()  # 36
    measures[CT.LOG_SUM_OF_FRO_OVER_MARGIN] = math.log(
        d) + (1/d) * (measures[CT.LOG_PROD_OF_FRO] - 2 * margin.log())  # 38
    measures[CT.LOG_SUM_OF_FRO] = math.log(
        d) + (1/d) * measures[CT.LOG_PROD_OF_FRO]  # 39

    print("Distance to Initialization")
    measures[CT.FRO_DIST] = dist_fro_norms.sum()  # 40
    measures[CT.DIST_SPEC_INIT] = dist_spec_norms.sum()  # 41
    measures[CT.PARAM_NORM] = fro_norms.sum()  # 42

    print("Path-norm")
    # Adapted from https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py#L98

    def _path_norm(model: ExperimentBaseModel) -> Tensor:
        model = deepcopy(model)
        model.eval()
        for param in model.parameters():
            if param.requires_grad:
                param.data.pow_(2)
        x = torch.ones([1] + list(model.dataset_type.D), device=device)
        x = model(x)
        del model
        return x.sum()

    @torch.no_grad()
    def _get_margin_loss(model, dataloader, L):
        # Calculate the first term in R.H.S. of Theorem D.1. [Daniel Roy 2017]
        assert loss != LossType.MSE, "MSE loss margin need modification!"
        margin_loss = 0.0
        for data, target in dataloader:
            logits = model(data).squeeze(-1)
            target_pn1 = 2 * target - 1
            margin_loss += torch.sum(
                    torch.maximum(torch.minimum(1 - L * torch.mul(target_pn1, logits),
                        torch.ones_like(logits)), torch.zeros_like(logits)))
        margin_loss = margin_loss / len(dataloader)
        return margin_loss

    @torch.no_grad()
    def _path_norm_bound(path_norm, model, L_grid, dataloader=trainNtest_loaders[0]):
        delta = 0.01
        model = deepcopy(model)
        model.eval()
        if model_type != ModelType.FCN:
            raise NotImplementedError("Path Norm bound only works for ReLU FCNs")
        # Rademacher complexity on the level set, when max data norm is 1
        R = 2.**(len(model.width_tuple)+1) * path_norm * torch.sqrt(
                torch.log(torch.tensor(2.*model.input_dim))/m)

        bound = torch.tensor(float("Inf"))
        for L in L_grid:
            bound_L = _get_margin_loss(model, dataloader, L) + 2*L*R + torch.sqrt(
                    torch.log(torch.tensor(2/delta))/(2*m))
            bound = torch.min(bound_L, bound)
            # print("L:", L)
            # print("Bound at L:", bound_L.item())
        return bound

    measures[CT.PATH_NORM] = _path_norm(model)  # 44
    measures[CT.PATH_NORM_OVER_MARGIN] = measures[CT.PATH_NORM] / \
        margin ** 2  # 43

    if hparams.center_data == True:
        raise NotImplementedError("Now the max of maximum norm of data samples is not 1!")
    elif model_type == ModelType.FCN:
        L_grid = np.arange(0.1, 1, 0.01)
        measures[CT.PATH_NORM_BOUND] = _path_norm_bound(measures[CT.PATH_NORM], model, L_grid)


    print("Flatness-based measures")
    sigma = _pacbayes_sigma(model, trainNtest_loaders[0], acc, seed)

    def _pacbayes_bound(reference_vec: Tensor) -> Tensor:
        return (reference_vec.norm(p=2) ** 2) / (4 * sigma ** 2) + math.log(m / sigma) + 10
    measures[CT.PACBAYES_INIT] = _pacbayes_bound(dist_w_vec)  # 48
    measures[CT.PACBAYES_ORIG] = _pacbayes_bound(w_vec)  # 49
    measures[CT.PACBAYES_FLATNESS] = torch.tensor(1 / sigma ** 2)  # 53

    print("Magnitude-aware Perturbation Bounds")
    mag_eps = 1e-3
    mag_sigma = _pacbayes_sigma(model, trainNtest_loaders[0], acc, seed, mag_eps)
    omega = num_params

    def _pacbayes_mag_bound(reference_vec: Tensor) -> Tensor:
        numerator = mag_eps ** 2 + \
            (mag_sigma ** 2 + 1) * (reference_vec.norm(p=2)**2) / omega
        denominator = mag_eps ** 2 + mag_sigma ** 2 * dist_w_vec ** 2
        return 1/4 * (numerator / denominator).log().sum() + math.log(m / mag_sigma) + 10
    measures[CT.PACBAYES_MAG_INIT] = _pacbayes_mag_bound(dist_w_vec)  # 56
    measures[CT.PACBAYES_MAG_ORIG] = _pacbayes_mag_bound(w_vec)  # 57
    measures[CT.PACBAYES_MAG_FLATNESS] = torch.tensor(1 / mag_sigma ** 2)  # 61

    # All stuff below are optimzing the PAC-Bayes bound,
    # Codes are adapted from https://github.com/gkdziugaite/pacbayes-opt/blob/master/
    # Bear in mind that this whole get_all_measures function works inside @torch.no_grad()

    def get_weights_and_biases(model: ExperimentBaseModel) -> List[Tensor]:
        blacklist = {'bn'}
        return [p for name, p in model.named_parameters() if all(x not in name for x in blacklist)]

    def KLdiv(pbar,p):
        return pbar * np.log(pbar/p) + (1-pbar) * np.log((1-pbar)/(1-p))

    def KLdiv_prime(pbar,p):
        return (1-pbar)/(1-p) - pbar/p

    def Newt(p,q,c):
        newp = p - (KLdiv(q,p) - c)/KLdiv_prime(q,p)
        return newp

    def approximate_BPAC_bound(q, c, niter=5):
        # Newton's method to find the numerical approximation for KL^{-1}(q|c)
        b_tilda = q + np.sqrt(c/2)
        if b_tilda >= 1.0:
            return 1.0
        for i in range(niter):
            b_tilda = Newt(b_tilda, q, c)
        return b_tilda

    def evaluate_acc_stoch_and_det(model,
            log_post_std_list,
            rng,
            dataloader=trainNtest_loaders[0]):
        model.eval()
        num_to_evaluate_on = len(dataloader.dataset)
        num_correct_stoch = 0
        num_correct_det = 0

        for data, target in dataloader:
            logits_det = model(data).squeeze(-1)
            pred_det = logits_det.data > 0
            batch_correct_det = pred_det.eq(target.data.view_as(
                pred_det)).type(torch.FloatTensor).cpu()
            num_correct_det += batch_correct_det.sum()

            logits_stoch = FCN_with_noise(
                    model, log_post_std_list, rng, data).squeeze(-1)
            pred_stoch = logits_stoch.data > 0
            batch_correct_stoch = pred_stoch.eq(target.data.view_as(
                pred_stoch)).type(torch.FloatTensor).cpu()
            num_correct_stoch += batch_correct_stoch.sum()
        acc_det = num_correct_det.item() / num_to_evaluate_on
        acc_stoch = num_correct_stoch.item() / num_to_evaluate_on
        return(acc_det, acc_stoch)

    @torch.no_grad()
    def evaluate_acc_stoch(model,
            log_post_std_list,
            rng,
            dataloader=trainNtest_loaders[0]):
        # Only evaluate SNN accuracy, as you don't need to evaluate det DNN accuracy
        # for every iteration in N_SNN_samples.
        model.eval()
        num_to_evaluate_on = len(dataloader.dataset)
        num_correct_stoch = 0

        for data, target in dataloader:
            logits_stoch = FCN_with_noise(
                    model, log_post_std_list, rng, data).squeeze(-1)
            pred_stoch = logits_stoch.data > 0
            batch_correct_stoch = pred_stoch.eq(target.data.view_as(
                pred_stoch)).type(torch.FloatTensor).cpu()
            num_correct_stoch += batch_correct_stoch.sum()
        acc_stoch = num_correct_stoch.item() / num_to_evaluate_on
        return(acc_stoch)

    def PACB_objective(current_weights, log_prior_std,
            log_post_std_list, prior_weights, nparams):

        delta = torch.tensor(0.025)
        b = torch.tensor(100.)
        c = torch.tensor(0.1) # These variables are defined according to original paper
        norm_post_variance = torch.sum(torch.stack(list(map(
            lambda x: torch.sum(torch.exp(x*2)), log_post_std_list))))
        norm_params = torch.sum(torch.stack(list(map(
            lambda x,y: torch.sum((x-y)**2), current_weights, prior_weights))))
        sum_log_post_variance = 2*torch.sum(torch.stack(list(map(
            lambda x: torch.sum(x), log_post_std_list))))
        mean_weights_component = (norm_params) / (torch.exp(2*log_prior_std))
        var_weights_component = norm_post_variance/(torch.exp(
            2*log_prior_std)) - sum_log_post_variance + 2*nparams*log_prior_std
        KLdivTimes2 = mean_weights_component + var_weights_component - nparams
        factor1 = 2*torch.log(b)
        factor2 = 2*torch.log(torch.maximum(
            torch.log(c) - 2 * log_prior_std, torch.tensor(1e-2)))
        Bquad = KLdivTimes2/2 + torch.log(np.pi**2*m/(6*delta))+factor1+factor2
        B_RE = Bquad / (m-1)

        return norm_post_variance,norm_params,sum_log_post_variance,factor1,factor2,B_RE

    @torch.no_grad()
    def evaluate_final_bound(model,
            log_prior_std,
            log_post_std_list,
            rng,
            prior_weights,
            dataloader,
            nparams,
            N_SNN_samples=1,
            twice_bound=False):

        b = torch.tensor(100.)
        c = torch.tensor(0.1) # These variables are defined according to original paper
        init_log_prior_std = log_prior_std
        current_weights = get_weights_and_biases(model)
        jdisc = b * (torch.log(c) - 2 * init_log_prior_std)
        jdisc_up = torch.ceil(jdisc)
        jdisc_down = torch.floor(jdisc)
        init_log_prior_std_up = (torch.log(c) - jdisc_up / b) / 2
        init_log_prior_std_down = (torch.log(c) - jdisc_down / b) / 2

        _,_,_,_,_,B_RE_up = PACB_objective(current_weights,
                init_log_prior_std_up, log_post_std_list, prior_weights, nparams)
        _,_,_,_,_,B_RE_down = PACB_objective(current_weights,
                init_log_prior_std_down, log_post_std_list, prior_weights, nparams)
        B_RE_optimal = torch.minimum(B_RE_up, B_RE_down).cpu().numpy()

        mean_train_acc_stoch = 0
        mean_test_acc_stoch = 0
        for ns in range(N_SNN_samples):
            mean_train_acc_stoch += evaluate_acc_stoch(
                    model, log_post_std_list, rng, trainNtest_loaders[0])
            mean_test_acc_stoch += evaluate_acc_stoch(
                    model, log_post_std_list, rng, trainNtest_loaders[1])
        mean_train_acc_stoch = mean_train_acc_stoch / N_SNN_samples
        mean_test_acc_stoch = mean_test_acc_stoch / N_SNN_samples
        train_acc_det, _ = evaluate_acc_stoch_and_det(
                model, log_post_std_list, rng, dataloader=trainNtest_loaders[0])
        test_acc_det, _ = evaluate_acc_stoch_and_det(
                model, log_post_std_list, rng, dataloader=trainNtest_loaders[1])


        # Technically, here we need to use 
        # approximate_BPAC_bound twice. Once for the SNN (N_SNN_samples)
        # MC approximation and once for the actual bound approximation.
        # But in practice if you set a large N_SNN_samples, the gap between
        # MC approximation of SNN training error and the upper bound is very small
        # Considering setting a large N_SNN_samples is very time-consuming,
        # here we follow the original code and only do it once.
        if twice_bound == True:
            SNN_MC_upper_bound = approximate_BPAC_bound(
                    1-mean_train_acc_stoch, np.log(2/0.01)/N_SNN_samples)
            print("SNN_MC_upper_bound: ", SNN_MC_upper_bound)
            bpac = approximate_BPAC_bound(SNN_MC_upper_bound, B_RE_optimal)
            print("Optimized PAC-Bayes bound (with confidence 0.965):", bpac)
        else:
            bpac = approximate_BPAC_bound(1-mean_train_acc_stoch, B_RE_optimal)
            print("Optimized PAC-Bayes bound:", bpac)
        return bpac, B_RE_optimal, mean_train_acc_stoch, mean_test_acc_stoch, train_acc_det, test_acc_det


    @torch.enable_grad()
    def PACB_optimization(model,
            prior_weights,
            epochs = 200,
            device=device,
            seed=seed,
            loss=LossType.CE,
            training_set=trainNtest_loaders[0].dataset):
        if loss != LossType.CE:
            raise NotImplementedError

        model = deepcopy(model)
        weights_and_biases_list = get_weights_and_biases(model)


        w_b_vec = get_vec_params(weights_and_biases_list)
        nparams = len(w_b_vec)

        log_prior_std = torch.tensor(-3.0, requires_grad=True)
        log_post_std_list = []
        for _w in weights_and_biases_list:
            # log_post_std = torch.log(2*torch.abs(_w).detach().clone()).requires_grad_(True)   #Orinigal code

            # log_post_std = torch.abs(_w.clone().detach()).requires_grad_(True) 
            # Psudo code. Practically really bad.
            temp = torch.abs(_w.clone().detach())
            _w_cutoff = 1.e-6*torch.ones_like(temp)
            log_post_std = (0.5 * torch.log(torch.where(
                temp > 1.e-6, temp, _w_cutoff))).requires_grad_(True)
            # # Text in the paper. 
            # log_post_std = torch.tensor(0.5*torch.log(torch.abs(_w)), requires_grad=True)
            log_post_std_list.append(log_post_std)

        trainable_params = list(model.parameters())
        trainable_params.append(log_prior_std)
        trainable_params = trainable_params + log_post_std_list

        optimizer = torch.optim.RMSprop(trainable_params, lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=150000, gamma=0.1)

        rng = torch.Generator()
        rng.manual_seed(7472358837 + seed)

        dataloader = torch.utils.data.DataLoader(
            training_set,
            batch_size=100,
            shuffle=True,
            num_workers=0)
        model.train()
        for epoch_idx in range(epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                optimizer.zero_grad()
                current_weights = get_weights_and_biases(model)

                logits = FCN_with_noise(model, log_post_std_list, rng, data).squeeze(-1)
                if loss == LossType.CE:
                    cross_entropy = F.binary_cross_entropy_with_logits(logits, target)
                else:
                    raise NotImplementedError
                (norm_post_variance,norm_params,sum_log_post_variance,
                        factor1,factor2,B_RE) = PACB_objective(
                                current_weights, log_prior_std,
                                log_post_std_list, prior_weights, nparams)

                cost = cross_entropy + torch.sqrt(B_RE/2)
                cost.backward()
                optimizer.step()
                scheduler.step()

            if (epoch_idx + 1) % 100 == 0:
                train_acc_det, train_acc_stoch = evaluate_acc_stoch_and_det(
                        model, log_post_std_list, rng)
                print("Train_acc_det: %.5f, Train_acc_stoch: %.5f" %
                        (train_acc_det, train_acc_stoch) +
                        " | B_RE term: %.5f, cost: %.5f" %
                        (B_RE.item(), cost.item()) +
                        " log_prior_std: %.5f" % log_prior_std.data +
                        " norm_post_variance: %.5f" % norm_post_variance.data +
                        " norm_params: %.5f" % norm_params.data +
                        " sum_log_post_variance: %.5f" %sum_log_post_variance.data +
                        " factor1: %.5f, factor2: %.5f" %(factor1.data,factor2.data) +
                        "\n"+"*"*100
                        )

        bpac, B_RE_optimal, mean_train_acc_stoch, mean_test_acc_stoch, train_acc_det, test_acc_det = evaluate_final_bound(model,log_prior_std,log_post_std_list,rng,
                prior_weights,dataloader,nparams,N_SNN_samples=1,twice_bound=False)

        return bpac, B_RE_optimal, mean_train_acc_stoch, mean_test_acc_stoch, train_acc_det, test_acc_det


    if optimize_PAC_Bayes_bound:
        print("PAC-Bayes bound optimization")
        prior_weights = get_weights_and_biases(init_model)
        prior_weights = [p.clone().detach().to(device) for p in prior_weights]
        bpac, B_RE_optimal, mean_train_acc_stoch, mean_test_acc_stoch, train_acc_det, test_acc_det = PACB_optimization(
                model, prior_weights,epochs=int(50000/m*400))
        #bpac, B_RE_optimal, mean_train_acc_stoch, mean_test_acc_stoch, train_acc_det, test_acc_det = PACB_optimization(
        #        model, prior_weights,epochs=int(1))
        # The number of epochs here is trying to be in line with the paper.

        print("mean_train_acc_stoch: %.5f, mean_test_acc_stoch: %.5f, train_acc_det: %.5f, test_acc_det: %.5f" %(
            mean_train_acc_stoch, mean_test_acc_stoch, train_acc_det, test_acc_det))
        measures[CT.BPAC_OPT] = torch.tensor(bpac, device=device, dtype=torch.float32)
        measures[CT.B_RE_OPTIMAL] = torch.tensor(B_RE_optimal, device=device, dtype=torch.float32)
        # Note, whether the following "measures" are complexity measures are highly debatable
        # I put them all in "measures" because they are byproducts of calculating real
        # generalization measures. They are accuracies of stochastic/deterministic DNNs
        # AFTER PAC-Bayes optimization. So they are NOT the original SGD-trained DNN.
        measures[CT.MEAN_TRAIN_ACC_STOCH] = torch.tensor(mean_train_acc_stoch, device=device, dtype=torch.float32)
        measures[CT.MEAN_TEST_ACC_STOCH] = torch.tensor(mean_test_acc_stoch, device=device, dtype=torch.float32)
        measures[CT.TRAIN_ACC_DET] = torch.tensor(train_acc_det, device=device, dtype=torch.float32)
        measures[CT.TEST_ACC_DET] = torch.tensor(test_acc_det, device=device, dtype=torch.float32)


    # Adjust for dataset size
    def adjust_measure(measure: CT, value: float) -> float:
        if measure.name.startswith('LOG_'):
            return 0.5 * (value - np.log(m))
        elif measure.name in ['PRIOR', 'MAR_LIK', 'PRIOR_MC',
                              'PRIOR_ELBO', 'MAR_LIK_MC', 'MAR_LIK_ELBO',
                              "BPAC_OPT", "B_RE_OPTIMAL", "MEAN_TRAIN_ACC_STOCH",
                              "MEAN_TEST_ACC_STOCH", "TRAIN_ACC_DET",
                              "TEST_ACC_DET","PATH_NORM_BOUND"]:
            return value
        else:
            return np.sqrt(value / m)
    return {k: adjust_measure(k, v.item()) for k, v in measures.items()}
