import pandas as pd


def slash2dot(string):
    if '/' in string:
        new_str = string[:string.index('/')]+'.'+string[string.index('/')+1:]
        return new_str
    else:
        return string

def format_data(csv_file):

    data = pd.read_csv(csv_file)
    data = data.drop(columns='Name')
    data = data.rename(str.lower, axis='columns')
    data = data.rename(slash2dot, axis='columns')
    data = data.drop(data[data.state == 'running'].index)
    data = data.drop(columns=['state','group', 'created',
        'updated', 'end time', 'runtime', 'epochs', 'use_cuda'])

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

    data['hp.dataset'] = data['hp.dataset'].str.replace(
            'cifar10_binary','cifar-binary')
    data['hp.dataset'] = data['hp.dataset'].str.replace(
            'svhn_binary','svhn-binary')


    data.to_csv('formatted.csv',index=False)


if __name__ == '__main__':
    format_data('180_runs.csv')
