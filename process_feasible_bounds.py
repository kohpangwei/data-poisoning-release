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


### Parameters
verbose = True
use_bias = True

tol = 1e-5
num_iter_to_throw_out = 0
learning_rate = 0.1

print_interval = 1000
percentile = 70

dataset_num_iter_after_burnin = {
    'imdb': 6000, #12000,
    'enron': 8000,
    'dogfish': 15000,
    'mnist_17': 8000
}

dataset_learning_rates = {
    'imdb': 0.001,
    'enron': 0.1,
    'dogfish': 0.05, 
    'mnist_17': 0.1
}
###

# By default, we generate upper and lower bounds for the given dataset,
# assuming a fixed oracle sphere+slab defense without any integrity constraints.
# If we pass in any of the boolean flags,
# it switches instead to processing already-generated upper bounds 
# and attacks (lower bounds) to put them in the same format as the default 
# upper and lower bounds generated here, so that we can easily compare the results.

# Load and make sure parameters match
parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', help='One of: imdb, enron, dogfish, mnist_17')
parser.add_argument('--slab', action='store_true', help='Data-dependent attack')
parser.add_argument('--grad', action='store_true', help='Gradient-based attack baseline')
parser.add_argument('--labelflip', action='store_true', help='Label flip attack baseline')
parser.add_argument('--int', action='store_true', help='Integer-constrained attack')
parser.add_argument('--percentile', type=float)

args = parser.parse_args()

if args.percentile is not None:
    percentile = args.percentile

dataset_name = args.dataset_name
process_slab = args.slab
process_grad = args.grad
process_labelflip = args.labelflip
process_int = args.int
assert (process_slab + process_grad + process_labelflip + process_int) <= 1
if process_slab + process_grad + process_labelflip + process_int == 0:
    no_process = True
else:
    no_process = False
X_train, Y_train, X_test, Y_test = datasets.load_dataset(dataset_name)

assert dataset_name in datasets.DATASET_WEIGHT_DECAYS
epsilons = datasets.DATASET_EPSILONS[dataset_name]
norm_sq_constraint = datasets.DATASET_NORM_SQ_CONSTRAINTS[dataset_name]

learning_rate = dataset_learning_rates[dataset_name]
num_iter_after_burnin = dataset_num_iter_after_burnin[dataset_name]

upper_total_losses = np.zeros_like(epsilons)
upper_good_losses = np.zeros_like(epsilons)
upper_bad_losses = np.zeros_like(epsilons)
upper_good_acc = np.zeros_like(epsilons)
upper_bad_acc = np.zeros_like(epsilons)
upper_params_norm_sq = np.zeros_like(epsilons)

lower_total_train_losses = np.zeros_like(epsilons)
lower_avg_good_train_losses = np.zeros_like(epsilons)
lower_avg_bad_train_losses = np.zeros_like(epsilons)
lower_test_losses = np.zeros_like(epsilons)
lower_overall_train_acc = np.zeros_like(epsilons)
lower_good_train_acc = np.zeros_like(epsilons)
lower_bad_train_acc = np.zeros_like(epsilons)
lower_test_acc = np.zeros_like(epsilons)
lower_params_norm_sq = np.zeros_like(epsilons)
lower_weight_decays = np.zeros_like(epsilons)

### Initial training on clean data
print('=== Training on clean data ===')

# Special case for the imdb dataset: 
# We set the initial guess for the correct weight_decay 
# to avoid unnecessary computation, since it takes a bit of time
# to binary search to find this
if (no_process) and (dataset_name == 'imdb'):
    clean_weight_decay = 0.0102181799337
else if not no_process:
    standard_f = np.load(datasets.get_bounds_path(dataset_name, norm_sq_constraint))
    clean_weight_decay = standard_f['lower_weight_decays'][0]
    assert np.all(epsilons == standard_f['epsilons'])
else:
    clean_weight_decay = None

train_loss, train_acc, test_loss, test_acc, \
  params_norm_sq, weight_decay, orig_params, orig_bias = \
  upper_bounds.svm_with_rho_squared(
    X_train, Y_train, 
    X_test, Y_test, 
    norm_sq_constraint, 
    use_bias=use_bias, 
    weight_decay=clean_weight_decay)


if epsilons[0] == 0:

    upper_total_losses[0] = train_loss
    upper_good_losses[0] = train_loss
    upper_bad_losses[0] = 0
    upper_good_acc[0] = train_acc
    upper_bad_acc[0] = 0
    upper_params_norm_sq[0] = params_norm_sq

    lower_total_train_losses[0] = train_loss
    lower_avg_good_train_losses[0] = train_loss
    lower_avg_bad_train_losses[0] = 0
    lower_test_losses[0] = test_loss
    lower_overall_train_acc[0] = train_acc
    lower_good_train_acc[0] = train_acc
    lower_bad_train_acc[0] = 0
    lower_test_acc[0] = test_acc
    lower_params_norm_sq[0] = params_norm_sq
    lower_weight_decays[0] = weight_decay

# Init guess
lower_weight_decay = weight_decay

if no_process:
    # If we want to add ignore_slab, here's the place to do it
    minimizer = upper_bounds.Minimizer()

    # This is a hack that's needed because we subsequently
    # do randomized rounding on the attack points, which
    # pushes stuff out of the feasible set, so we need to 
    # set the percentile to be some conservative low amount
    if (dataset_name == 'imdb') and (percentile == 70):
        class_map, centroids, centroid_vec, sphere_radii, _ = data.get_data_params(
            X_train, Y_train, percentile=15)
        _, _, _, _, slab_radii = data.get_data_params(
            X_train, Y_train, percentile=65)
    else:
        class_map, centroids, centroid_vec, sphere_radii, slab_radii = data.get_data_params(X_train, Y_train, percentile=percentile)
    max_iter = num_iter_after_burnin + num_iter_to_throw_out
    needed_iter = int(np.round(np.max(epsilons) * X_train.shape[0]) + num_iter_to_throw_out)
    assert max_iter >= needed_iter, 'Not enough samples; increase max_iter to at least %s.' % needed_iter


for epsilon_idx, epsilon in enumerate(epsilons):

    if epsilon == 0:
        continue

    print('=== epsilon = %s ===' % epsilon)

    # Generate our normal upper/lower bound
    if no_process:
        init_w = np.zeros_like(orig_params)
        init_b = 0
        X_modified, Y_modified, idx_train, idx_poison = generate_upper_and_lower_bounds(
            X_train, Y_train,
            norm_sq_constraint, 
            epsilon,
            max_iter,
            num_iter_to_throw_out,
            learning_rate,
            init_w, init_b,
            class_map, centroids, centroid_vec, 
            sphere_radii, slab_radii,
            minimizer,
            verbose=verbose,
            print_interval=print_interval)

    elif process_slab:
        if dataset_name == 'dogfish':
            f = sio.loadmat(datasets.get_slab_mat_path(dataset_name, epsilon, percentile=50))
        elif dataset_name == 'mnist_17':
            f = sio.loadmat(datasets.get_slab_mat_path(dataset_name, epsilon))
        metadata_final = f['metadata_final']        
        int_upper_good_loss = metadata_final[0][0][0][0][0]
        int_upper_bad_loss = metadata_final[0][0][1][0][0]
        int_upper_good_acc = metadata_final[0][0][2][0][0]
        int_upper_bad_acc = None
        int_upper_norm_theta = metadata_final[0][0][3][0][0]
        int_upper_bias = metadata_final[0][0][4][0][0]     
        assert f['epsilon'][0][0] == epsilon

        int_upper_params_norm_sq = int_upper_norm_theta ** 2 + int_upper_bias ** 2
        int_upper_total_loss = int_upper_good_loss + epsilon * int_upper_bad_loss

        upper_total_losses[epsilon_idx] = int_upper_total_loss
        upper_good_losses[epsilon_idx] = int_upper_good_loss
        upper_bad_losses[epsilon_idx] = int_upper_bad_loss
        upper_good_acc[epsilon_idx] = int_upper_good_acc
        upper_bad_acc[epsilon_idx] = int_upper_bad_acc
        upper_params_norm_sq[epsilon_idx] = int_upper_params_norm_sq

        print('Final upper bound:')
        print("  Total loss (w/o reg)   : %s" % int_upper_total_loss)
            
        if int_upper_params_norm_sq > norm_sq_constraint:
            print('*********************************************************')
            print('* WARNING: params_norm_sq (%s) > norm_sq_constraint (%s)' % (int_upper_params_norm_sq, norm_sq_constraint))
            print('*********************************************************')

        X_poison = f['bestX'][0, ...] 
        Y_poison = f['bestY'][0, ...].reshape(-1)       
        X_modified, Y_modified, idx_train, idx_poison = process_matlab_train_test(
            f, 
            X_train, Y_train,
            X_test, Y_train,
            X_poison, Y_poison)
    
    elif process_grad:
        # Upper bound is not valid for grad attack
        weight_decay = standard_f['lower_weight_decays'][epsilon_idx]
        print(standard_f['lower_weight_decays'])
        # This is super hacky
        # and a consequence of weight_decay changing slightly for mnist before we started running gradient descent...
        # Actual weight decays are
        # [ 0.05666385  0.03622478  0.04557456  0.06812952  0.08532804  0.08444606 0.08047717  0.07430335  0.06812952]
        # TODO: Fix after deadline
        if dataset_name == 'mnist_17':
            weight_decay = [None, 0.0347366333008, 0.0455780029297, 0.068100810051, 0.0852910876274, 0.0848503112793, 0.0804425477982, 0.0742716789246, 0.068100810051][epsilon_idx]

        # Actual weight decays are
        # [ 0.00100763  0.0091314   0.03627361  0.08111179  0.10680558  0.11184358 0.10075999  0.09471439  0.08967639]
        elif dataset_name == 'dogfish':
            weight_decay = [None, 0.00815894421645, 0.0363878186312, 0.0813030943666, 0.106897273836, 0.105893580523, 0.100875113961, 0.0943511074294, 0.0888307942105][epsilon_idx]

        X_modified, Y_modified, X_test2, Y_test2, idx_train, idx_poison = datasets.load_attack_npz(
            dataset_name, 
            datasets.get_grad_attack_wd_npz_path(dataset_name, epsilon, weight_decay),
            take_path=True)
        
        assert np.all(np.isclose(X_test2, X_test))
        assert np.all(Y_test2 == Y_test)
        assert np.all(np.isclose(X_modified[idx_train, :], X_train)) 
        assert np.all(Y_train == Y_modified[idx_train])

    elif process_labelflip:
        X_modified, Y_modified, X_test2, Y_test2, idx_train, idx_poison = datasets.load_attack_npz(
            dataset_name, 
            datasets.get_labelflip_attack_npz_filename(dataset_name, epsilon, norm_sq_constraint=None))   
        datasets.check_poisoned_data(X_train, Y_train, X_modified[idx_poison, :], Y_modified[idx_poison], X_modified, Y_modified)

    else:
        # Upper bound is not valid for feasible attack
        # int_upper_good_loss = metadata_final[0][0][0][0][0]
        # int_upper_bad_loss = metadata_final[0][0][1][0][0]
        # int_upper_good_acc = metadata_final[0][0][2][0][0]
        # int_upper_bad_acc = metadata_final[0][0][3][0][0]
        # int_upper_norm_theta = metadata_final[0][0][4][0][0]
        # int_upper_bias = metadata_final[0][0][5][0][0]

        # HARDCODE WARNING
        if dataset_name == 'imdb':
            X_modified, Y_modified, X_test2, Y_test2, idx_train, idx_poison = datasets.load_attack_npz(
                dataset_name, 
                datasets.get_int_attack_npz_filename(dataset_name, epsilon, norm_sq_constraint, percentile=15.0))
            assert (X_test2 - X_test).nnz == 0
            assert np.all(Y_test2 == Y_test)
            assert (X_modified[idx_train, :] - X_train).nnz == 0
            assert np.all(Y_train == Y_modified[idx_train])

        elif dataset_name == 'enron':
            f = sio.loadmat(datasets.get_int_mat_path(dataset_name, epsilon))
            assert f['epsilon'][0][0] == epsilon

            if sparse.issparse(f['X_train']):
                assert np.all(f['X_train'].toarray() == X_train)
                assert np.all(f['X_test'].toarray() == X_test)
            else:
                assert np.all(f['X_train'] == X_train)
                assert np.all(f['X_test'] == X_test)
            
            assert np.all(f['y_train'].reshape(-1) == Y_train)        
            assert np.all(f['y_test'].reshape(-1) == Y_test)

            X_poison = f['X_pert'] # This is not stored as a sparse matrix
            Y_poison = f['y_pert'].reshape(-1)

            X_modified, Y_modified, idx_train, idx_poison = process_matlab_train_test(
                f, 
                X_train, Y_train,
                X_test, Y_train,
                X_poison, Y_poison)
        else: 
            raise ValueError, 'invalid dataset'

    
    total_train_loss, avg_good_train_loss, avg_bad_train_loss, test_loss, \
      overall_train_acc, good_train_acc, bad_train_acc, test_acc, \
      params_norm_sq, lower_weight_decay = upper_bounds.evaluate_attack(
        X_modified, Y_modified, 
        X_test, Y_test, 
        idx_train, idx_poison, 
        epsilon, 
        lower_weight_decay,
        norm_sq_constraint,        
        use_bias)

    lower_total_train_losses[epsilon_idx] = total_train_loss
    lower_avg_good_train_losses[epsilon_idx] = avg_good_train_loss
    lower_avg_bad_train_losses[epsilon_idx] = avg_bad_train_loss
    lower_test_losses[epsilon_idx] = test_loss
    lower_overall_train_acc[epsilon_idx] = overall_train_acc
    lower_good_train_acc[epsilon_idx] = good_train_acc
    lower_bad_train_acc[epsilon_idx] = bad_train_acc
    lower_test_acc[epsilon_idx] = test_acc
    lower_params_norm_sq[epsilon_idx] = params_norm_sq
    lower_weight_decays[epsilon_idx] = lower_weight_decay

    print('** WARNING: Only looking at top one...')
    # if process_slab:        
    #     for k in range(1, f['bestX'].shape[0]):
    #         X_poison = f['bestX'][k, ...] 
    #         Y_poison = f['bestY'][k, ...].reshape(-1)

    #         X_modified = np.concatenate((X_train, X_poison), axis=0)
    #         Y_modified = np.concatenate((Y_train, Y_poison), axis=0)
    #         idx_train = slice(0, X_train.shape[0])
    #         idx_poison = slice(X_train.shape[0], X_modified.shape[0])
            
    #         total_train_loss, avg_good_train_loss, avg_bad_train_loss, test_loss, \
    #           overall_train_acc, good_train_acc, bad_train_acc, test_acc, \
    #           params_norm_sq, lower_weight_decay = upper_bounds.evaluate_attack(
    #             X_modified, Y_modified, 
    #             X_test, Y_test, 
    #             idx_train, idx_poison, 
    #             epsilon, 
    #             lower_weight_decay,
    #             norm_sq_constraint,
    #             use_bias)

    #         if lower_avg_good_train_losses[epsilon_idx] < avg_good_train_loss:
    #             lower_total_train_losses[epsilon_idx] = total_train_loss
    #             lower_avg_good_train_losses[epsilon_idx] = avg_good_train_loss
    #             lower_avg_bad_train_losses[epsilon_idx] = avg_bad_train_loss
    #             lower_test_losses[epsilon_idx] = test_loss
    #             lower_overall_train_acc[epsilon_idx] = overall_train_acc
    #             lower_good_train_acc[epsilon_idx] = good_train_acc
    #             lower_bad_train_acc[epsilon_idx] = bad_train_acc
    #             lower_test_acc[epsilon_idx] = test_acc
    #             lower_params_norm_sq[epsilon_idx] = params_norm_sq
    #             lower_weight_decays[epsilon_idx] = lower_weight_decay

    attack_save_path = None
    if dataset_name in ['dogfish', 'mnist_17', 'enron']:
        if no_process:
            attack_save_path = datasets.get_attack_npz_path(dataset_name, epsilon, norm_sq_constraint, percentile)
        elif process_slab:
            attack_save_path = datasets.get_slab_attack_npz_path(dataset_name, epsilon, norm_sq_constraint)
        elif process_grad:
            attack_save_path = datasets.get_grad_attack_npz_path(dataset_name, epsilon, norm_sq_constraint)
        elif process_labelflip:
            attack_save_path = datasets.get_labelflip_attack_npz_path(dataset_name, epsilon, norm_sq_constraint)            
        elif process_int:            
            attack_save_path = datasets.get_int_attack_npz_path(dataset_name, epsilon, norm_sq_constraint)        

    # We generate the imdb data without integrity constraints
    # and then do the randomized rounding after
    # so we need a separate call to this script with the --int flag
    # to fully process its results.
    # To save space, we don't save it to disk if it's processing slab/grad/etc.
    elif (dataset_name in ['imdb']) and (no_process):
        X_poison_sparse = sparse.csr_matrix(data.rround(data.threshold(X_modified[idx_poison, :])))
        X_modified = sparse.vstack((X_train, X_poison_sparse))
        attack_save_path = datasets.get_int_attack_npz_path(dataset_name, epsilon, norm_sq_constraint, percentile)            

    if attack_save_path is not None:
        np.savez(
            attack_save_path,
            X_modified=X_modified,
            Y_modified=Y_modified,
            X_test=X_test,
            Y_test=Y_test,
            idx_train=idx_train,
            idx_poison=idx_poison
            ) 

if no_process:
    bounds_save_path = datasets.get_bounds_path(dataset_name, norm_sq_constraint, percentile)
elif process_slab:
    bounds_save_path = datasets.get_slab_bounds_path(dataset_name, norm_sq_constraint)
elif process_grad:
    bounds_save_path = datasets.get_grad_bounds_path(dataset_name, norm_sq_constraint)
elif process_labelflip:
    bounds_save_path = datasets.get_labelflip_bounds_path(dataset_name, norm_sq_constraint)      
elif process_int:
    bounds_save_path = datasets.get_int_bounds_path(dataset_name, norm_sq_constraint)


np.savez(
    bounds_save_path,
    percentile=percentile,
    weight_decay=weight_decay,
    epsilons=epsilons,
    upper_total_losses=upper_total_losses,
    upper_good_losses=upper_good_losses,
    upper_bad_losses=upper_bad_losses,
    upper_good_acc=upper_good_acc,
    upper_bad_acc=upper_bad_acc,
    upper_params_norm_sq=upper_params_norm_sq,
    lower_total_train_losses=lower_total_train_losses,
    lower_avg_good_train_losses=lower_avg_good_train_losses,
    lower_avg_bad_train_losses=lower_avg_bad_train_losses,
    lower_test_losses=lower_test_losses,
    lower_overall_train_acc=lower_overall_train_acc,
    lower_good_train_acc=lower_good_train_acc,
    lower_bad_train_acc=lower_bad_train_acc,
    lower_test_acc=lower_test_acc,
    lower_params_norm_sq=lower_params_norm_sq,
    lower_weight_decays=lower_weight_decays     
    )

