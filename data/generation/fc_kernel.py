import torch
import numpy as np
import os, sys, time, random


def kern(X1: torch.tensor, X2: torch.tensor,
        number_layers: int, sigmaw: float, sigmab: float):
    """
    Calculate
    X1: training set or part of the training set
        shape: (m, input_dim) or (n_max, input_dim)
        for n_max = 10000

        When m > n_max, X1 and X2 are different slices of
        the training set.
        K_symm = kern(X1, X1)
        K_cross = kern(X1, X2)
        and the full kernel looks like:
                       |
          K_symm(X1,X1)| K_cross(X1,X2)
                       |
            - - - - - - - - - - -
                       |
                       |
               K_cross | K_symm(X2,X2)
                       |

    X1 and X2 can have different size.

    sigmab, sigmaw: float

    Return: Kernel matrix of shape(N1, N2)
    """
    N1 = X1.shape[0] # number of training samples in X1
    input_dim = X1.shape[1]

    if X2 == None:
        # setting X2==None is the same as X2=X1.
        # But perhaps for the sake of saving RAM

        # Gram matrix of all K0(x, x') in X1
        K = sigmab**2 + sigmaw**2 * torch.matmul(X1, torch.t(X1))/input_dim
        for l in range(number_layers):
            K_diag = torch.diagonal(K).unsqueeze(-1)
            K1 = torch.tile(K_diag, (1, N1))
            K2 = torch.tile(torch.t(K_diag), (N1, 1))

            K12 = torch.mul(K1, K2) # elementwise multiplication
            costheta = torch.div(K, torch.sqrt(K12))
            theta = torch.acos(costheta)
            K = sigmab**2 + ((sigmaw**2/(2*np.pi)) * np.sqrt(K12) *
                            (torch.sin(theta) + (np.pi - theta) * costheta))
        return K

    else:
        N2 = X2.shape[0]
        K = sigmab**2 + sigmaw**2 * torch.matmul(X1, torch.t(X2))/input_dim
        K1_diag = sigmab**2 + sigmaw**2 * torch.sum(torch.mul(X1,X1), axis=1, keepdims=True)/input_dim
        K2_diag = sigmab**2 + sigmaw**2 * torch.sum(torch.mul(X2,X2), axis=1, keepdims=True)/input_dim
        for l in range(number_layers):
            K1 = torch.tile(K1_diag, (1, N2))
            K2 = torch.tile(torch.t(K2_diag), (N1, 1))

            K12 = torch.mul(K1, K2) # elementwise multiplication
            costheta = torch.div(K, torch.sqrt(K12))
            theta = torch.acos(costheta)
            K = sigmab**2 + ((sigmaw**2/(2*np.pi)) * np.sqrt(K12) *
                            (torch.sin(theta) + (np.pi - theta) * costheta))
            K1_diag = sigmab**2 + (sigmaw**2 / 2) * K1_diag
            K2_diag = sigmab**2 + (sigmaw**2 / 2) * K2_diag

        return K


def kernel_matrix(X: torch.tensor, number_layers: int,
        sigmaw: float, sigmab: float, n_gpus: int = 1):

    # the definition of n_gpus here is really unnecessary,
    # considering hydra don't really allow simple gpu sharing.
    # One motherboard only has one piece of GPU. Sadge.
    """
    X: All samples. In the case of calculating marginal likelihood
       it's the training set. In calculating the volume it's the
       concatenation of training set and test set.
    """

    m = X.shape[0]
    n_max = min(10000, m)

    # Creating slices of the final kernel matrix
    # if m=20000 and n_max=10000, the slices woulu be
    # [((0,10000), (0,10000)),
    #  ((0,10000), (10000,20000)),
    #  ((10000,20000), (10000,20000))]
    slices = []
    if m <= n_max:
        slices.append((slice(0, m), slice(0, m)))
    else:
        for j in range(0, m-n_max+1, n_max):
            for i in range(j, m-n_max+1, n_max):
                slices.append((slice(j, j+n_max), slice(i, i+n_max)))
            if m % n_max != 0:
                # There are last bits
                slices.append((slice(j, j+n_max), slice(i+n_max, i+n_max+m%n_max)))
        if m % n_max != 0:
            slices.append((slice(j+n_max, j+n_max+m%n_max), slice(i+n_max, i+n_max+m%n_max)))


    K_matrix = np.zeros((m, m), dtype=np.float64)

    for j_s, i_s in slices:
        if j_s == i_s:
            X1 = X[j_s]
            K_symm = kern(X1, None, number_layers, sigmaw, sigmab)
            K_matrix[j_s, i_s] = K_symm
        else:
            X1 = X[j_s]
            X2 = X[i_s]
            K_cross = kern(X1, X2, number_layers, sigmaw, sigmab)
            K_matrix[j_s, i_s] = K_cross
            K_matrix[i_s, j_s] = K_cross.T

    return K_matrix


def kernel_matrix_cross(X_train: torch.tensor,
        X_test: torch.tensor, number_layers: int, sigmaw: float, sigmab: float):
    # This function is for calculating posteriors of individual sample in the testset.
    # They will all be 1-D Gaussian, ignoring the covariance between testset samples.
    # Should improve performance (O(m^2) to O(m))
    #
    # Returns: K(X, X*) of size (m_train, m_test)
    m1 = X_train.shape[0]
    m2 = X_test.shape[0]
    m = max(m1, m2)
    n_max = min(10000, m)

    if m <= n_max:
        K_cross_mtx = np.zeros((m1, m2), dtype=np.float64)
        K_cross_mtx[:, :] = kern(X_train, X_test, number_layers, sigmaw, sigmab)

    else:
        slices = []
        if min(m1, m2) <= n_max:
            if m1 > m2: # test set size smaller or equal to 10000
                for j in range(0, m-n_max+1, n_max):
                    slices.append((slice(j, j+n_max), slice(0, m2)))
                if m % n_max != 0:
                    slices.append((slice(j+n_max, j+n_max+m%n_max), slice(0, m2)))
            else:
                for j in range(0, m-n_max+1, n_max):
                    slices.append((slice(0, m1), slice(j, j+n_max)))
                if m % n_max != 0:
                    slices.append((slice(0, m1), slice(j+n_max, j+n_max+m%n_max)))
        else: # both m1 and m2 larger than n_max=10000
            for j in range(0, m1-n_max+1, n_max):
                for i in range(0, m2-n_max+1, n_max):
                    slices.append((slice(j, j+n_max), slice(i, i+n_max)))
                if m2 % n_max != 0 :
                    slices.append((slice(j, j+n_max), slice(i+n_max, i+n_max+m2%n_max)))
            if m1 % n_max != 0:
                for i in range(0, m2-n_max+1, n_max):
                    slices.append((slice(j+n_max, j+n_max+m1%n_max), slice(i, i+n_max)))
                if m2 % n_max != 0 :
                    slices.append((slice(j+n_max, j+n_max+m1%n_max), slice(i+n_max, i+n_max+m2%n_max)))

        K_cross_mtx = np.zeros((m1, m2), dtype=np.float64)
        for j_s, i_s in slices:
            X1 = X_train[j_s]
            X2 = X_test[i_s]
            K_cross_block = kern(X1, X2, number_layers, sigmaw, sigmab)
            K_cross_mtx[j_s, i_s] = K_cross_block

    return K_cross_mtx


def k_diag_vector(X: torch.tensor, number_layers: int, sigmaw: float, sigmab: float):
    # return the digonal elements (self-variance) of all samples in the dataset X
    # K(x, x) for all x in the dataset set.
    # As a vector of the same length as dataset set size.
    # this is most useful when getting the predictive posterior on the test set,
    # when the covariance can be safely ignored.

    N = X.shape[0]
    input_dim = X.shape[1]

    K_diag = sigmab**2 + sigmaw**2 * torch.sum(torch.mul(X,X), axis=1, keepdims=True)/input_dim

    for l in range(number_layers):
        K_diag = sigmab**2 + (sigmaw**2 / 2) * K_diag

    assert len(K_diag.shape) == 2 and K_diag.shape[1] == 1

    return np.array(K_diag.squeeze(1), dtype=np.float64)



def kernel_matrix_wo_testset_covariance(X_train: torch.tensor,
        X_test: torch.tensor, number_layers: int, sigmaw: float, sigmab: float):
    raise NotImplementedError


if __name__ == '__main__':
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
    from .experiment_config import HParams as hparams
    from .experiment_config import Config as config
    from .models import NiN_binary, FCN
    from .dataset_helpers import get_dataloaders

    device = 'cuda' if hparams.use_cuda else 'cpu'
    model = FCN([1024,1024], hparams.dataset_type)
    (_,train_eval_loader,_) = get_dataloaders(hparams, config, device)
    (xs_train, ys_train) = _get_xs_ys_from_dataset(train_eval_loader.dataset)

    if xs_train.is_cuda: xs_train = xs_train.cpu()
    K = kernel_matrix(xs_train, model.number_layers, np.sqrt(2), 0)
    #print(K.shape)

    np.save('K_anal.npy', K)

















