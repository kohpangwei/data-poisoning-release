from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster, metrics, svm, model_selection

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse

import data_utils as data

def print_cv_results(results):
    print('Cross-validation results:')
    for idx, weight_decay in enumerate(results['weight_decays']):
        print('  weight_decay = %s    mean CV acc (modified) = %.3f    mean CV acc (clean) = %.3f' % (
            weight_decay, 
            results['mean_cv_scores_modified'][idx],
            results['mean_cv_scores_train'][idx]))
    print('Best weight_decay (modified data): %s' % results['best_weight_decay_modified'])
    print('Best weight_decay (clean data): %s' % results['best_weight_decay_train'])


def print_test_results(results):

    print('Training on the clean data:')
    print('  Train acc (clean) : ', results['test']['clean']['train_acc_clean'])
    print('  Train loss (clean): ', results['test']['clean']['train_loss_clean'])
    print('  Test acc          : ', results['test']['clean']['test_acc'])
    print('  Test loss         : ', results['test']['clean']['test_loss'])
    print('  Norm of params: ', results['test']['clean']['params_norm'])
    print('')

    print('Training on the poisoned data:')
    print('  Train acc (overall) : ', results['test']['modified']['train_acc_overall'])
    print('  Train acc (clean)   : ', results['test']['modified']['train_acc_clean'])
    print('  Train acc (poison)  : ', results['test']['modified']['train_acc_poison'])
    print('  Train loss (overall): ', results['test']['modified']['train_loss_overall'])
    print('  Train loss (clean)  : ', results['test']['modified']['train_loss_clean'])
    print('  Train loss (poison) : ', results['test']['modified']['train_loss_poison'])
    print('  Test acc            : ', results['test']['modified']['test_acc'])
    print('  Test loss           : ', results['test']['modified']['test_loss'])
    print('  Norm of params: ', results['test']['modified']['params_norm'])
    print('')

    diff_in_test_acc = results['test']['clean']['test_acc'] - results['test']['modified']['test_acc']
    print('Difference in test accuracy between clean and poisoned: %.3f' % diff_in_test_acc)
    print('epsilon = %.3f' % results['epsilon'])
    print('Leverage: %.3f' % results['test']['leverage'])


def summarize_defs(defense_labels, defense_labels_no_emp, results):

    for use_emp_label in ['true', 'emp']:
        for defense_label in defense_labels:
            defense_emp_label = '%s_%s' % (defense_label, use_emp_label)
            print('%s == ' % '{:20}'.format(defense_emp_label), end='')

            if defense_emp_label in results:
                print('frac_to_remove: %.2f      test acc: %.2f      leverage: %.2f' % (
                    results[defense_emp_label]['cv_frac_to_remove'],
                    results[defense_emp_label]['cv_test_acc'],
                    results[defense_emp_label]['leverage']))
            else:
                print('<not run>')
        print('')
    
    for defense_label in defense_labels_no_emp:
        print('%s == ' % '{:20}'.format(defense_label), end='')

        if defense_label in results:
            print('frac_to_remove: %.2f      test acc: %.2f      leverage: %.2f' % (
                results[defense_label]['cv_frac_to_remove'],
                results[defense_label]['cv_test_acc'],
                results[defense_label]['leverage']))
        else:
            print('<not run>')        

    print('')


def plot_dists(datadef, defense_label, results, all_dists, no_emp=False, num_bins=100):

    for use_emp_label in ['true', 'emp']:
        
        if no_emp:
            defense_emp_label = defense_label
        else:
            print('Using %s class centroids:' % use_emp_label)            
            defense_emp_label = '%s_%s' % (defense_label, use_emp_label)

        if defense_emp_label in results:

            idx = results['dist_labels'].index(defense_emp_label)
            dists = all_dists[:, idx]

            datadef.plot_dists(    
                dists,
                num_bins=num_bins)

            f, axs = plt.subplots(1, 2, figsize=(12, 6))
            fracs_to_remove = results[defense_emp_label]['fracs_to_remove']
            axs[0].plot(fracs_to_remove, results[defense_emp_label]['val_accs'], label='Validation')
            axs[0].plot(fracs_to_remove, results[defense_emp_label]['test_accs'], label='Test')
            axs[0].set_xlabel('Fraction of data removed')
            axs[0].set_ylabel('Accuracy')
            axs[0].legend()

            axs[1].plot(fracs_to_remove, results[defense_emp_label]['fracs_of_good_points_kept'], c='green', label='Good data')
            axs[1].plot(fracs_to_remove, results[defense_emp_label]['fracs_of_bad_points_kept'], c='red', label='Bad data')
            axs[1].set_xlabel('Fraction of overall data removed')
            axs[1].set_ylabel('Fraction of data removed')
            axs[1].legend()

            plt.show()

        else:
            print('Defense %s not run.' % defense_emp_label)

        if no_emp:
            break