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

    sigmab, sigmaw: float
    """
    N = X1.shape[0] # number of training samples in this slice
    input_dim = X1.shape[1]

    if X2 == None:
        # setting X2==None is the same as X2=X1.
        # But perhaps for the sake of saving RAM

        # Gram matrix of all K0(x, x') in X1
        K = sigmab**2 + sigmaw**2 * torch.matmul(X1, torch.t(X1))/input_dim
        for l in range(number_layers):
            K_diag = torch.diagonal(K).unsqueeze(-1)
            K1 = torch.tile(K_diag, (1, N))
            K2 = torch.tile(torch.t(K_diag), (N, 1))

            K12 = torch.mul(K1, K2) # elementwise multiplication
            costheta = torch.div(K, torch.sqrt(K12))
            theta = torch.acos(costheta)
            K = sigmab**2 + (sigmaw**2/(2*np.pi)) * tf.sqrt(K12) *
                            (torch.sin(theta) + (np.pi - theta) * costheta)
        return K

    else:
        K = sigmab**2 + sigmaw**2 * torch.matmul(X1, torch.t(X2))/input_dim
        K1_diag = sigmab**2 + sigmaw**2 * torch.sum(torch.mul(X1,X1), axis=1, keepdims=True)/input_dim
        K2_diag = sigmab**2 + sigmaw**2 * torch.sum(torch.mul(X2,X2), axis=1, keepdims=True)/input_dim
        for l in range(number_layers):
            K1 = torch.tile(K1_diag, (1, N))
            K2 = torch.tile(torch.t(K2_diag), (N, 1))

            K12 = torch.mul(K1, K2) # elementwise multiplication
            costheta = torch.div(K, torch.sqrt(K12))
            theta = torch.acos(costheta)
            K = sigmab**2 + (sigmaw**2/(2*np.pi)) * tf.sqrt(K12) *
                            (torch.sin(theta) + (np.pi - theta) * costheta)
            K1_diag = sigmab**2 + (sigmaw**2 / 2) * K1_diag
            K2_diag = sigmab**2 + (sigmaw**2 / 2) * K2_diag

        return K


def kernel_matrix(X: np.array, number_layers: int,
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
    # Adjust n_max so it's evenly split
    for i in range(n_max, 0, -1):
        if m % i == 0:
            n_max = i
            break

    # Creating slices of the final kernel matrix
    # if m=20000 and n_max=10000, the slices woulu be
    # [((0,10000), (0,10000)),
    #  ((0,10000), (10000,20000)),
    #  ((10000,20000), (10000,20000))]
    slices = list((slice(j, j+n_max), slice(i, i+n_max))
                    for j in range(0, m, n_max)
                    for i in range(j, m, n_max))

    K_matrix = np.zeros((m, m), dtype=np.float64)

    for j_s, i_s in slices[j]:
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






























