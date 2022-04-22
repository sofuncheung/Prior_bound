import pandas as pd
import os
import numpy as np


def slash2dot(string):
    if '/' in string:
        new_str = string[:string.index('/')]+'.'+string[string.index('/')+1:]
        return new_str
    else:
        return string

def average_by_group(data):
    '''
    NOT WORKING!!!
    averageing all columns if two experiments record belong to the same group
    (meaning they only differ by random seeds)
    '''
    raise NotImplementedError()

    group_list = set(data['group'])
    df_temp_list = []
    for g in group_list:
        rows_of_same_group = []
        for _, row in data.iterrows():
            if row['group'] == g:
                rows_of_same_group.append(row)
        print(rows_of_same_group)
        df_temp = pd.concat(rows_of_same_group, axis=0, ignore_index=True)
        print(df_temp)
        df_temp_averaged = df_temp.mean()
        de_temp_list.append(df_temp_averaged)

    return(pd.concat(df_temp_list, axis=0, ignore_index=True))


def format_data(csv_file):

    data = pd.read_csv(csv_file)
    data = data.drop(columns='Name')
    data = data.rename(str.lower, axis='columns')
    data = data.rename(slash2dot, axis='columns')
    data = data.drop(data[data.state == 'running'].index)
    to_be_dropped = ['state', 'created',
            'updated', 'end time', 'runtime', 'epochs', 'use_cuda',
            'notes', 'user', 'tags', 'runtime', 'sweep',
            'ce_target', 'ce_target_milestones', 'data_seed',
            'optimizer_type', 'id', 'job type', 'hostname',
            'description','commit','github', 'gpu count', 'gpu type'
            ]
    for i in to_be_dropped:
        try:
            data = data.drop(columns=i)
        except:
            pass

    rename_dic = {'cross_entropy.train': 'cross_entropy',
            'accuracy.train': 'gen.train_acc',
            'accuracy.test': 'gen.val_acc',
            'dataset_type': 'hp.dataset',
            'lr': 'hp.lr',
            'model_depth': 'hp.model_depth',
            'model_width': 'hp.model_width'
            }
    data = data.rename(columns = rename_dic)
    data = data.drop(columns=['complexity.l2', 'complexity.l2_dist'])
    data['hp.dataset'] = data['hp.dataset'].str.lower()

    data['hp.train_dataset_size'] = data['train_dataset_size']
    data['gen.gap'] = data['gen.train_acc'] - data['gen.val_acc']
    data['is.converged'] = data['cross_entropy'] < 0.01
    data['is.high_train_accuracy'] = data['gen.train_acc'] > 0.99

    data['is.full_train_accuracy'] = data['gen.train_acc'] == 1

    data['hp.dataset'] = data['hp.dataset'].str.replace(
            'cifar10_binary','cifar10-binary')
    data['hp.dataset'] = data['hp.dataset'].str.replace(
            'svhn_binary','svhn-binary')
    data['hp.dataset'] = data['hp.dataset'].str.replace(
            'mnist_binary','mnist-binary')
    data['hp.dataset'] = data['hp.dataset'].str.replace(
            'fashionmnist_binary','fashionmnist-binary')

    # Rescale mar_lik for the actual bound in Theorem 5.1 of
    # https://arxiv.org/abs/2012.04115
    data["mar_lik_bound"] = (data['complexity.mar_lik'] * data["hp.train_dataset_size"] / np.log10(np.e)
                             + np.log(data["hp.train_dataset_size"]) + 2*np.log(100)) / (
                                 data["hp.train_dataset_size"] - 1)
    data["prior_bound"] = (data['complexity.prior'] * data["hp.train_dataset_size"]
                           / np.log10(np.e) + np.log(100)) / data["hp.train_dataset_size"]

    data.to_csv('formatted.csv',index=False)


if __name__ == '__main__':
    wandb_raw_files = [f for f in os.listdir() if f.startswith('wandb_export_')]
    if len(wandb_raw_files) != 1:
        raise FileNotFoundError()
    else:
        wandb_raw_file = wandb_raw_files[0]

    format_data(wandb_raw_file)
