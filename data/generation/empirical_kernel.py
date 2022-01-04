import numpy as np
import os, sys, time, random
import gc
import torch
import torch.nn as nn


def empirical_K(model, data, number_samples, device, seed,
        n_gpus=1,
        empirical_kernel_batch_size=5000,
        truncated_init_dist=False,
        store_partial_kernel=False, # True will not average the kernel at thn end.
        partial_kernel_n_proc=1,
        partial_kernel_index=0):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Here model is the NIN defined as in paper
    # "In Search of Robust Measures of Generalization"
    # with an additional subtraction between the two logits in the last layer

    #number_samples = data.shape[0] # Number of MC samples
    num_tasks = number_samples

    if store_partial_kernel:
        size = partial_kernel_n_proc
        rank = partial_kernel_index
        num_tasks_per_job = num_tasks//size
        tasks = list(range(int(rank*num_tasks_per_job),int((rank+1)*num_tasks_per_job)))

        if rank < num_tasks%size:
            tasks.append(size*num_tasks_per_job+rank)
    else:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        num_tasks_per_job = num_tasks//size
        tasks = list(range(int(rank*num_tasks_per_job),int((rank+1)*num_tasks_per_job)))
        # ID of tasks un each process.
        if rank < num_tasks%size:
            tasks.append(size*num_tasks_per_job+rank)
        # if there's x tasks left, then give it evenly to the first x-rank processes.

    print("Doing process %d of %d" % (rank, size))

    m = len(data)
    if device == 'cuda':
        covs = torch.zeros([m, m], dtype=torch.float16).to(device)
    else:
        covs = np.zeros((m,m), dtype=np.float16)
    local_index = 0 # index of task in a particular precess
    update_chunk = 10000 # Guillermo use 10000
    num_chunks = covs.shape[0]//update_chunk
    print("num_chunks: ", num_chunks)

    for index in tasks:
        start_time = time.time()
        if index % 100 == 0:
            print("sample for kernel", index)

        if local_index > 0:
            model.apply(weight_reset)
        # Guillermo chose to use Keras model, while I do everything in Pytorch.
        # The principle is in every MC sample the weights and biases are chosen from 
        # He-normal distribution (In lots of GP papers they use forward-version of Xavier
        # but for He-normal you only need to change the 'gain' from 1 to sqrt{2}. The term
        # 'gain' is used in Pytorch docs: https://pytorch.org/docs/stable/nn.init.html)

        # Also, Guillermo's 'reset_weights' will re-initialize weights and biases but keep
        # BatchNorm layer unchanged. While in Pytorch case the default initialization for 
        # BatchNorm is constant already.

        X = model_predict(model, data,
                min(empirical_kernel_batch_size, len(data)), device)
        # X is the SINGLE logits in this set of prior bound comparison experiments
        if device == 'cuda':
            if len(X.shape) == 1:
                X.unsqueeze_(1)
            if covs.shape[0] > update_chunk:
                for i in range(num_chunks):
                    covs[i*update_chunk:(i+1)*update_chunk] += (
                            (1/X.shape[1]) * torch.matmul(
                                X[i*update_chunk:(i+1) * update_chunk], X.T))
                last_bits = slice(update_chunk*num_chunks,covs.shape[0])
                covs[last_bits] += (
                        (1/X.shape[1]) *
                        torch.matmul(X[last_bits], X.T))
            else:
                covs += (1 / X.shape[1]) * torch.matmul(X,X.T)
        else: # On cpu and use numpy
            if len(X.shape) == 1:
                X = np.expand_dims(X, 1)
            if covs.shape[0] > update_chunk:
                for i in range(num_chunks):
                    covs[i*update_chunk:(i+1)*update_chunk] += (
                            (1/X.shape[1]) * np.matmul(
                                X[i*update_chunk:(i+1) * update_chunk], X.T))
                last_bits = slice(update_chunk*num_chunks,covs.shape[0])
                covs[last_bits] += (
                        (1/X.shape[1]) *
                        np.matmul(X[last_bits],X.T))
            else:
                covs += (1 / X.shape[1]) * np.matmul(X,X.T)
        sys.stdout.flush()
        local_index += 1
        gc.collect()
        if index % 100 == 0:
            print("--- %s seconds ---" % (time.time() - start_time))

    if size > 1 and not store_partial_kernel:
        covs1_recv = None
        covs2_recv = None
        if rank == 0: # Do following in the first (main) process
            if device == 'cuda':
                covs1_recv = torch.zeros_like(covs[:25000,:], dtype=torch.float16)
                covs2_recv = torch.zeros_like(covs[25000:,:], dtype=torch.float16)
            else:
                covs1_recv = np.zeros_like(covs[:25000,:], dtype=np.float16)
                covs2_recv = np.zeros_like(covs[25000:,:], dtype=np.float16)
        #print(covs[25000:,:])
        comm.Reduce(covs[:25000,:], covs1_recv, op=MPI.SUM, root=0)
        comm.Reduce(covs[25000:,:], covs2_recv, op=MPI.SUM, root=0)

        if rank == 0:
            if device == 'cuda':
                covs_recv = torch.cat([covs1_recv,covs2_recv],0)
            else:
                covs_recv = np.concatenate([covs1_recv,covs2_recv],0)
            return covs_recv/number_samples
        else:
            return None
    else:
        #if covs.shape[0] > update_chunk:
        #    #make matrix symmetric
        #    covs = np.maximum(covs,covs.trasnpose())
        if store_partial_kernel:
            return covs
        else:
            return covs/number_samples


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
    if isinstance(m, nn.BatchNorm2d):
        pass
    # I have checked that upon different initializations the four
    # parameters in BatchNorm layers:
    # running_mean, running_var, weight, bias
    # are all the same. So no need to reset the parameters for
    # BatchNorm layers.



def model_predict(model, data, batch_size, device):
    r'''
    Get the output of Pytorch model in a multi-batch fashion,
    for memory saving purpose.

    Note: data here is actually a torch.utils.data.Dataset,
          but onlt it's images matter.
    '''

    model = model.to(device)
    model.eval()
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)
    outputs_list = []
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            outputs_list.append(outputs)

    return torch.cat(outputs_list, axis=0)



if __name__ == '__main__':
    from .experiment_config import HParams as hparams
    from .experiment_config import Config as config
    from .models import NiN_binary
    from .dataset_helpers import get_dataloaders
    device = 'cuda' if hparams.use_cuda else 'cpu'
    model = NiN_binary(hparams.model_depth, hparams.model_width,
                hparams.base_width, hparams.dataset_type)
    (_,train_eval_loader,_) = get_dataloaders(hparams, config, device)
    K = empirical_K(model, train_eval_loader.dataset, 10, device, hparams.seed,
            n_gpus=1,
            empirical_kernel_batch_size=256,
            truncated_init_dist=False,
            store_partial_kernel=False,
            partial_kernel_n_proc=1,
            partial_kernel_index=0,
            )
    print(K.shape)

