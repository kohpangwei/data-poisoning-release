from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import os 
import shutil
from subprocess import call
import numpy as np
import scipy.sparse as sparse
import scipy.io as sio
import IPython

import datasets

epsilon = 0.1
random_seed = 1

dataset_best_iters = {
    'dogfish': 4000,
    'mnist_17': 1240,
    'enron': 1560
}

for dataset_name in ['dogfish', 'mnist_17', 'enron', 'imdb']:
    weight_decay = datasets.DATASET_WEIGHT_DECAYS[dataset_name]

    output_root = os.path.join(datasets.DATA_FOLDER, 'influence_data')

    attack_filename ='%s_labelflip_eps-%s_rs-%s.npz' % (dataset_name, epsilon, random_seed)
    shutil.copyfile(        
        os.path.join(output_root, attack_filename),
        os.path.join(datasets.DATA_FOLDER, dataset_name, attack_filename))

    call(
        'python test_defenses.py %s %s --no_defense --weight_decay %s' % (dataset_name, attack_filename, weight_decay),
        shell=True)

    if dataset_name in dataset_best_iters:
        
        attack_filename = 'smooth_hinge_%s_sphere-True_slab-True_start-copy_lflip-True_step-0.001_t-0_rs-%s_x_iter-%s.npz' % (
            dataset_name, random_seed, dataset_best_iters[dataset_name])
    
        shutil.copyfile(        
            os.path.join(output_root, attack_filename),
            os.path.join(datasets.DATA_FOLDER, dataset_name, attack_filename))

        call(
            'python test_defenses.py %s %s --no_defense --weight_decay %s' % (dataset_name, attack_filename, weight_decay),
            shell=True)