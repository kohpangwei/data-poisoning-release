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

stop_after = 5

print_interval = 500
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

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', help='One of: imdb, enron, dogfish, mnist_17')
parser.add_argument('--ignore_slab', action="store_true")
parser.add_argument('--percentile', type=float)
args = parser.parse_args()

dataset_name = args.dataset_name
ignore_slab = args.ignore_slab
if args.percentile is not None:
    percentile = args.percentile

X_train, Y_train, X_test, Y_test = datasets.load_dataset(dataset_name)

epsilons = datasets.DATASET_EPSILONS[dataset_name]
learning_rate = dataset_learning_rates[dataset_name]
norm_sq_constraint = datasets.DATASET_NORM_SQ_CONSTRAINTS[dataset_name]

# if ignore_slab:
#     epsilons = [0.1]
# epsilons = [0, 0.1, 0.2, 0.3, 0.4]

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

print('=== Training on clean data ===')

# Special case for the imdb dataset: 
# We set the initial guess for the correct weight_decay 
# to avoid unnecessary computation, since it takes a bit of time
# to binary search to find this
if dataset_name == 'imdb':
    clean_weight_decay = 0.0102181799337
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

lower_weight_decay = weight_decay

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


# This is a hack that's needed because we subsequently
# do randomized rounding on the attack points, which
# pushes stuff out of the feasible set, so we need to 
# set the percentile to be some conservative low amount
if (dataset_name == 'imdb') and (percentile == 70):
    class_map, centroids, centroid_vec, sphere_radii, _ = data.get_data_params(
        X_train, Y_train, percentile=15)
    _, _, _, _, slab_radii = data.get_data_params(
        X_train, Y_train, percentile=60)
else:
    class_map, centroids, centroid_vec, sphere_radii, slab_radii = data.get_data_params(X_train, Y_train, percentile=percentile)
max_iter = num_iter_after_burnin + num_iter_to_throw_out
needed_iter = int(np.round(np.max(epsilons) * X_train.shape[0]) + num_iter_to_throw_out)
assert max_iter >= needed_iter, 'Not enough samples; increase max_iter to at least %s.' % needed_iter

minimizer = upper_bounds.Minimizer(use_slab = not(ignore_slab))


for epsilon_idx, epsilon in enumerate(epsilons):

    if epsilon == 0:
        continue

    old_w = None

    x_bs = np.zeros((max_iter, X_train.shape[1]))
    y_bs = np.zeros(max_iter)
    sum_w = np.zeros(X_train.shape[1])

    sum_of_grads_w_sq = np.ones(X_train.shape[1])
    sum_of_grads_w = np.zeros(X_train.shape[1])
    sum_of_grads_b = 0

    best_upper_bound = 10000

    current_lambda = 1 / learning_rate

    print('=== epsilon = %s ===' % epsilon)
    
    w = np.zeros_like(orig_params)
    b = 0

    stop_counter = 0

    for iter_idx in range(max_iter):

        grad_w, grad_b = hinge_grad(w, b, X_train, Y_train) 

        if verbose: 
            if iter_idx % print_interval == 0: print("At iter %s:" % iter_idx)    
        
        # Pick the y with the worse (more negative) margin        
        worst_margin = None
        for y_b in set(Y_train):

            class_idx = class_map[y_b]
            x_b = minimizer.minimize_over_feasible_set(
                y_b, 
                w, 
                centroids[class_idx, :], 
                centroid_vec, 
                sphere_radii[class_idx], 
                slab_radii[class_idx])

            margin = y_b * (w.dot(x_b) + b)
            if ((worst_margin is None) or (margin < worst_margin)):
                worst_margin = margin
                worst_y_b = y_b
                worst_x_b = x_b
                            
        # Take the gradient with respect to that y
        if worst_margin < 1:
            grad_w -= epsilon * worst_y_b * worst_x_b
            grad_b -= epsilon * worst_y_b
            
        bad_loss = hinge_loss(w, b, worst_x_b, worst_y_b)        
        
        # Store iterate to construct matching lower bound
        x_bs[iter_idx, :] = worst_x_b
        y_bs[iter_idx] = worst_y_b

        good_loss = hinge_loss(w, b, X_train, Y_train)
        params_norm_sq = (np.linalg.norm(w)**2 + b**2)
        total_loss = good_loss + epsilon * bad_loss
        
        if best_upper_bound > total_loss:
            best_upper_bound = total_loss            
            best_upper_good_loss = good_loss
            best_upper_bad_loss = bad_loss
            best_upper_params_norm_sq = params_norm_sq
            best_upper_good_acc = np.mean((Y_train * (X_train.dot(w) + b)) > 0)
            if worst_margin > 0:
                best_upper_bad_acc = 1.0
            else: 
                best_upper_bad_acc = 0.0            
            
        if verbose: 
            if iter_idx % print_interval == 0:
                print("  Bad margin (%s)         : %s" % (worst_y_b, worst_margin))
                print("  Bad loss (%s)           : %s" % (worst_y_b, bad_loss))                
                print("  Good loss               : %s" % good_loss)
                print("  Total loss              : %s" % total_loss)                
                print("  Sq norm of params_bias  : %s" % params_norm_sq)        
                print("  Grad w norm             : %s" % np.linalg.norm(grad_w))

        # Update outer minimization        
        sum_of_grads_w -= grad_w
        sum_of_grads_b -= grad_b

        candidate_lambda = np.sqrt(np.linalg.norm(sum_of_grads_w)**2 + sum_of_grads_b**2) / np.sqrt(norm_sq_constraint)
        if candidate_lambda > current_lambda:
            current_lambda = candidate_lambda            

        w = sum_of_grads_w / current_lambda
        b = sum_of_grads_b / current_lambda

    print('Optimization run for %s iterations' % max_iter)
    final_iter = iter_idx + 1
    # TODO: why do we need final_iter? Why not replace with max_iter?
    # We used to have stop_after and stop_counter but I think that is not used now.
    # TODO: isn't x_bs and y_bs always len max_iter? Why pass in an argument?

    upper_total_losses[epsilon_idx] = best_upper_bound
    upper_good_losses[epsilon_idx] = best_upper_good_loss
    upper_bad_losses[epsilon_idx] = best_upper_bad_loss
    upper_good_acc[epsilon_idx] = best_upper_good_acc
    upper_bad_acc[epsilon_idx] = best_upper_bad_acc
    upper_params_norm_sq[epsilon_idx] = best_upper_params_norm_sq

    print('Final upper bound:')
    print("  Total loss              : %s" % best_upper_bound)
    print('')

    num_clean = X_train.shape[0]

    # HACK
    np.savez(
        os.path.join(datasets.DATA_FOLDER, dataset_name, '%s_timelapse_eps-%s_normc-%s.npz' % (dataset_name, epsilon, norm_sq_constraint)),
        x_bs=x_bs,
        y_bs=y_bs)


    X_modified, Y_modified, idx_train, idx_poison = upper_bounds.sample_lower_bound_attack(
        X_train, Y_train, 
        x_bs, y_bs, 
        epsilon, 
        final_iter, num_iter_to_throw_out)
    
    total_train_loss, avg_good_train_loss, avg_bad_train_loss, test_loss, \
      overall_train_acc, good_train_acc, bad_train_acc, test_acc, \
      params_norm_sq, lower_weight_decay = upper_bounds.evaluate_attack(
        X_modified, Y_modified, 
        X_test, Y_test, 
        idx_train, idx_poison, 
        epsilon, 
        lower_weight_decay,
        # 15.5586,
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

    # Save attack points
    # I think we just need this for mnist and imdb for fig 1?
    # Presumably we don't save enron because we don't plot it
    if not ignore_slab:
        if dataset_name in ['imdb']:
            X_poison_sparse = sparse.csr_matrix(data.rround(data.threshold(X_modified[idx_poison, :])))
            X_modified = sparse.vstack((X_train, X_poison_sparse))
            save_path = datasets.get_int_attack_npz_path(dataset_name, epsilon, norm_sq_constraint, percentile)            
        else:
            save_path = datasets.get_attack_npz_path(dataset_name, epsilon, norm_sq_constraint, percentile)

        if dataset_name in ['dogfish', 'mnist_17', 'imdb']:
            np.savez(
                save_path,
                X_modified=X_modified,
                Y_modified=Y_modified,
                X_test=X_test,
                Y_test=Y_test,
                idx_train=idx_train,
                idx_poison=idx_poison
                )
        


if not ignore_slab:
    np.savez(
        datasets.get_bounds_path(dataset_name, norm_sq_constraint, percentile),
        percentile=percentile,
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

