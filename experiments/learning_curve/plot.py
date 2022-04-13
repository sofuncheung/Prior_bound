import matplotlib.pyplot as plt
import numpy as np
from common import get_complexity_measures, hoeffding_weight, get_hps, load_data, sign_error


DATA_PATH = "../../data/learning_curve/formatted.csv"
data = load_data(DATA_PATH)

svhn_data = data[data['hp.dataset']=='svhn-binary']
cifar10_data = data[data['hp.dataset']=='cifar10-binary']
mnist_data = data[data['hp.dataset']=='mnist-binary']
fashionmnist_data = data[data['hp.dataset']=='fashionmnist-binary']


all_data = [svhn_data, cifar10_data, mnist_data, fashionmnist_data]

c_measures = get_complexity_measures(data)
# Divide measures into different groups based on their category
VC_BOUNDS = ['complexity.params']
OUTPUT_MEASURES = ['complexity.inverse_margin']
SPECTRAL_BOUNDS = ['complexity.log_spec_init_main_fft',
                   'complexity.log_spec_orig_main_fft',
                   'complexity.log_prod_of_spec_over_margin_fft',
                   'complexity.log_prod_of_spec_fft',
                   'complexity.fro_over_spec_fft',
                   'complexity.dist_spec_init_fft',
                   'complexity.log_sum_of_spec_over_margin_fft',
                   'complexity.log_sum_of_spec_fft']
FROBENIUS_BOUNDS = ['complexity.log_prod_of_fro_over_margin',
                    'complexity.log_prod_of_fro',
                    'complexity.log_sum_of_fro_over_margin',
                    'complexity.log_sum_of_fro',
                    'complexity.fro_dist',
                    'complexity.param_norm']
PATH_NORM_MEASURES = ['complexity.path_norm',
                      'complexity.path_norm_over_margin']
FLATNESS_BOUNDS = ['complexity.pacbayes_init',
                   'complexity.pacbayes_orig',
                   'complexity.pacbayes_flatness',
                   'complexity.pacbayes_mag_init',
                   'complexity.pacbayes_mag_orig',
                   'complexity.pacbayes_mag_flatness']

GP_BOUNDS = ['complexity.mar_lik','complexity.prior']

all_bounds_category = [VC_BOUNDS,OUTPUT_MEASURES,SPECTRAL_BOUNDS,FROBENIUS_BOUNDS,
                       PATH_NORM_MEASURES,FLATNESS_BOUNDS,GP_BOUNDS]
# sanity check
# print(set(c_measures) -
#       set(OUTPUT_MEASURES) -
#       set(SPECTRAL_BOUNDS) -
#       set(FROBENIUS_BOUNDS)-
#       set(PATH_NORM_MEASURES)-
#       set(FLATNESS_BOUNDS)-
#       set(GP_BOUNDS))

for data in all_data:

    if data['hp.dataset'].iloc[0] == 'svhn-binary':
        DATA_NAME = 'svhn'
    elif data['hp.dataset'].iloc[0] == 'cifar10-binary':
        DATA_NAME = 'cifar10'
    elif data['hp.dataset'].iloc[0] == 'mnist-binary':
        DATA_NAME = 'mnist'
    elif data['hp.dataset'].iloc[0] == 'fashionmnist-binary':
        DATA_NAME = 'fashionmnist'



    for bounds in all_bounds_category:
        if 'complexity.params' in bounds:
            BOUNDS_NAME = 'VC'
        if 'complexity.inverse_margin' in bounds:
            BOUNDS_NAME = 'OUTPUT'
        if 'complexity.log_spec_init_main_fft' in bounds:
            BOUNDS_NAME = 'SPECTRAL'
        if 'complexity.log_prod_of_fro_over_margin' in bounds:
            BOUNDS_NAME = 'FROBENIUS_BOUNDS'
        if 'complexity.path_norm' in bounds:
            BOUNDS_NAME = 'PATH_NORM'
        if 'complexity.pacbayes_init' in bounds:
            BOUNDS_NAME = 'FLATNESS'
        if 'complexity.mar_lik' in bounds:
            BOUNDS_NAME = 'GP'
        fig, ax = plt.subplots()

        ax.plot(np.log10(data['train_dataset_size']),
                np.log10(data['generalization.error']),
                linestyle='dashed', label='test error')

        for c in bounds:
            adjust_data = data[c]
            if 'log' in c:
                adjust_data = np.exp(data[c])
            if 'mar_lik' in c or 'prior' in c:
                adjust_data = data[c] / np.log10(np.e)

            ax.plot(np.log10(data['train_dataset_size']),
                    np.log10(adjust_data), label=c.split('.')[1])

        ax.set_xlabel('Training set size',
                      fontdict={'fontsize': 20, 'fontweight': 'medium'})
        ax.set_ylabel('Generalization error',
                      fontdict={'fontsize': 20, 'fontweight': 'medium'})
        ax.set_xticks([2,3,4])
        ax.set_xticklabels([r'$10^2$', r'$10^3$', r'$10^4$'])
        ax.tick_params(direction='in')
        ax.legend()
        # fig.text(0.6,0.25,'FCN/MNIST/SGD', bbox=dict(facecolor='none'), fontsize=20)
        fig.savefig('%s_%s_bounds.png'%(DATA_NAME,BOUNDS_NAME), dpi=300)
        fig.clf()


