from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import os
import sys
import argparse
import json
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster, metrics, svm, model_selection

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse
import scipy.io as sio

import IPython

import data_utils as data
import datasets
import defenses
import defense_testers
import upper_bounds
from upper_bounds import hinge_loss, hinge_grad


for dataset_name in ['dogfish', 'mnist_17']:

    epsilons = datasets.DATASET_EPSILONS[dataset_name]
    norm_sq_constraint = datasets.DATASET_NORM_SQ_CONSTRAINTS[dataset_name]

    bounds_v2_path = os.path.join(
        datasets.DATA_FOLDER, 
        dataset_name, 
        '%s_slab_normc-%s_bounds_v2.npz' % (dataset_name, norm_sq_constraint))
    bounds_v3_path = os.path.join(
        datasets.DATA_FOLDER, 
        dataset_name, 
        '%s_slab_normc-%s_bounds_v3.npz' % (dataset_name, norm_sq_constraint))
    f_v2 = np.load(bounds_v2_path)
    f_v3 = np.load(bounds_v3_path)

    assert np.all(epsilons == f_v2['epsilons'])
    assert np.all(epsilons == f_v3['epsilons'])
    assert f_v2['percentile'] == f_v3['percentile']
    assert f_v2['percentile'] == 70
    percentile = f_v2['percentile']

    # Initialize with v3

    lower_test_losses = f_v3['lower_test_losses']
    lower_total_train_losses = f_v3['lower_total_train_losses']
    lower_test_acc = f_v3['lower_test_acc']
    lower_good_train_acc = f_v3['lower_good_train_acc']
    lower_bad_train_acc = f_v3['lower_bad_train_acc']
    lower_overall_train_acc = f_v3['lower_overall_train_acc']
    lower_avg_good_train_losses = f_v3['lower_avg_good_train_losses']
    lower_avg_bad_train_losses = f_v3['lower_avg_bad_train_losses']
    lower_params_norm_sq = f_v3['lower_params_norm_sq']
    lower_weight_decays = f_v3['lower_weight_decays']

    upper_total_losses = f_v3['upper_total_losses']
    upper_bad_losses = f_v3['upper_bad_losses']
    upper_good_losses = f_v3['upper_good_losses']
    upper_good_acc = f_v3['upper_good_acc']
    upper_bad_acc = f_v3['upper_bad_acc']
    upper_params_norm_sq = f_v3['upper_params_norm_sq'] 

    for epsilon_idx, epsilon in enumerate(epsilons):
        # Take lower upper bound
        if upper_total_losses[epsilon_idx] > f_v2['upper_total_losses'][epsilon_idx]:
            upper_total_losses[epsilon_idx] = f_v2['upper_total_losses'][epsilon_idx]
            upper_bad_losses[epsilon_idx] = f_v2['upper_bad_losses'][epsilon_idx]
            upper_good_losses[epsilon_idx] = f_v2['upper_good_losses'][epsilon_idx]
            upper_good_acc[epsilon_idx] = f_v2['upper_good_acc'][epsilon_idx]
            upper_bad_acc[epsilon_idx] = f_v2['upper_bad_acc'][epsilon_idx]
            upper_params_norm_sq[epsilon_idx] = f_v2['upper_params_norm_sq'][epsilon_idx]

        # Take higher lower bound based on lower_avg_good_train_losses
        if lower_avg_good_train_losses[epsilon_idx] < f_v2['lower_avg_good_train_losses'][epsilon_idx]:
            lower_test_losses[epsilon_idx] = f_v2['lower_test_losses'][epsilon_idx]
            lower_total_train_losses[epsilon_idx] = f_v2['lower_total_train_losses'][epsilon_idx]
            lower_test_acc[epsilon_idx] = f_v2['lower_test_acc'][epsilon_idx]
            lower_good_train_acc[epsilon_idx] = f_v2['lower_good_train_acc'][epsilon_idx]
            lower_bad_train_acc[epsilon_idx] = f_v2['lower_bad_train_acc'][epsilon_idx]
            lower_overall_train_acc[epsilon_idx] = f_v2['lower_overall_train_acc'][epsilon_idx]
            lower_avg_good_train_losses[epsilon_idx] = f_v2['lower_avg_good_train_losses'][epsilon_idx]
            lower_avg_bad_train_losses[epsilon_idx] = f_v2['lower_avg_bad_train_losses'][epsilon_idx]
            lower_params_norm_sq[epsilon_idx] = f_v2['lower_params_norm_sq'][epsilon_idx]
            lower_weight_decays[epsilon_idx] = f_v2['lower_weight_decays'][epsilon_idx]


        save_path = datasets.get_slab_bounds_path(dataset_name, norm_sq_constraint)


        np.savez(
            save_path,
            percentile=percentile,
            epsilons=epsilons,
            upper_total_losses=upper_total_losses,
            upper_good_losses=upper_good_losses,
            upper_bad_losses=upper_bad_losses,
            # upper_reg_losses=upper_reg_losses,
            upper_good_acc=upper_good_acc,
            upper_bad_acc=upper_bad_acc,
            upper_params_norm_sq=upper_params_norm_sq,
            lower_total_train_losses=lower_total_train_losses,
            lower_avg_good_train_losses=lower_avg_good_train_losses,
            lower_avg_bad_train_losses=lower_avg_bad_train_losses,
            lower_test_losses=lower_test_losses,
            # lower_reg_losses=lower_reg_losses,
            lower_overall_train_acc=lower_overall_train_acc,
            lower_good_train_acc=lower_good_train_acc,
            lower_bad_train_acc=lower_bad_train_acc,
            lower_test_acc=lower_test_acc,
            lower_params_norm_sq=lower_params_norm_sq,
            lower_weight_decays=lower_weight_decays     
            )
