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
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter

import data_utils as data
import datasets
import defenses
import defense_testers
from upper_bounds import hinge_loss, hinge_grad




parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', help='One of: imdb, enron, dogfish')
parser.add_argument('file_name', help='e.g., imdb_attack3.mat')
parser.add_argument('--min_leverage', help='Attack must initially exceed this leverage before defenses willl be run.', type=float)
parser.add_argument('--defense', help='Just test a single defense, e.g., centroid-vec, l2-ball, inverse-sigma, ...')
parser.add_argument('--no_defense', help='Skip all Q-based defenses.', action="store_true")
parser.add_argument('--weight_decay', help='If this is specified, the script will use this value instead of doing cross-validation.', type=float)
parser.add_argument("--debug", help="Changes parameters so that everything is sped up.", action="store_true")

args = parser.parse_args()

dataset_name = args.dataset_name
file_name = args.file_name
min_leverage = args.min_leverage
defense_to_test = args.defense
no_defense = args.no_defense
user_weight_decay = args.weight_decay
debug = args.debug

### Sanitize input and setup file paths
allowed_defenses = ['centroid-vec', 'l2-ball', 'l1-ball', 'inverse-sigma', 'pca-subspace', 'k-NN']
if ((defense_to_test is not None) and (defense_to_test not in allowed_defenses)):
    raise ValueError, 'defense must be in %s' % allowed_defenses 

# if ext != '.mat':
    # file_name = file_name + '.mat'    

attack_path = os.path.join(datasets.DATA_FOLDER, dataset_name, file_name)
if not os.path.isfile(attack_path):
    raise ValueError, 'Specified file %s does not exist' % attack_path

output_web_path = datasets.get_web_html_path(file_name)
output_ipynb_path = datasets.get_output_ipynb_path(dataset_name, file_name)
output_html_path = datasets.get_output_html_path(dataset_name, file_name)
output_mat_path = datasets.get_output_mat_path(dataset_name, file_name)
output_dists_path = datasets.get_output_dists_path(dataset_name, file_name)
output_json_path = datasets.get_output_json_path(dataset_name, file_name)

if os.path.isfile(output_html_path):
    print('There is already a report at %s. This will overwrite it.' % output_html_path)
    print('If you do not want this to happen, break out now!')
    print('')

if debug: print('### Debug mode.')
print('### Testing defenses against %s/%s.' % (dataset_name, file_name))
print('')



### Load data and setup

X_modified, Y_modified, X_test, Y_test, idx_train, idx_poison = datasets.load_attack(
    dataset_name,
    file_name)

X_train = X_modified[idx_train, :]
Y_train = Y_modified[idx_train]
X_poison = X_modified[idx_poison, :]
Y_poison = Y_modified[idx_poison]

results = defaultdict(dict)

datadef = defenses.DataDef(X_modified, Y_modified, X_test, Y_test, idx_train, idx_poison)
print('Number of good training examples: %s' % X_train.shape[0])
print('Number of poisoned training examples: %s' % X_poison.shape[0])
print('epsilon = %.3f' % datadef.epsilon)
print('\n')

results['epsilon'] = datadef.epsilon

if not debug:
    tol_svm = 1e-6

    random_state_svm = 24
    max_iter_cv_svm = 1000
    max_iter_test_svm = 4000
    weight_decays = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]

    num_folds = 5
    max_frac_to_remove = 0.30
    frac_increment = 0.05

    inverse_sigma_dim = 1000 # This only gets used if the number of features > 2048.
    num_neighbors = 5 # for k-NN

elif debug:
    tol_svm = 1e-6

    random_state_svm = 24
    max_iter_cv_svm = 10
    max_iter_test_svm = 40
    weight_decays = [0.1, 0.05]

    num_folds = 2
    max_frac_to_remove = 0.20
    frac_increment = 0.20

    inverse_sigma_dim = 10 
    num_neighbors = 2

if user_weight_decay is not None:
    weight_decays = [user_weight_decay]

all_dists = np.zeros((X_modified.shape[0], 0))
results['dist_labels'] = []

### Cross-validation
print('## Running %s-fold cross-validation to pick weight_decay' % num_folds)

k_fold = model_selection.KFold(n_splits=num_folds, shuffle=True, random_state=2)

best_cv_score_modified = -1
best_cv_score_train = -1
results['mean_cv_scores_modified'] = []
results['mean_cv_scores_train'] = []
for weight_decay in weight_decays: #, 0.001, 0.0005, 0.0001]:

    C = 1.0 / (X_modified.shape[0] * weight_decay)        
    svm_model = svm.LinearSVC(
        C=C,
        loss='hinge',
        tol=tol_svm,
        fit_intercept=True,
        random_state=random_state_svm,
        max_iter=max_iter_cv_svm)
    
    cv_scores_modified = model_selection.cross_val_score(
        svm_model, 
        X_modified, Y_modified, 
        cv=k_fold, 
        n_jobs=np.min((num_folds, 8)))

    C = 1.0 / (X_train.shape[0] * weight_decay)        
    svm_model = svm.LinearSVC(
        C=C,
        loss='hinge',
        tol=tol_svm,
        fit_intercept=True,
        random_state=random_state_svm,
        max_iter=max_iter_cv_svm)
    
    cv_scores_train = model_selection.cross_val_score(
        svm_model, 
        X_train, Y_train, 
        cv=k_fold, 
        n_jobs=np.min((num_folds, 8)))

    mean_cv_score_modified = np.mean(cv_scores_modified)
    mean_cv_score_train = np.mean(cv_scores_train)
    results['mean_cv_scores_modified'].append(mean_cv_score_modified)
    results['mean_cv_scores_train'].append(mean_cv_score_train)

    print('weight_decay = %s:' % weight_decay)
    print('  Val acc (modified data): %.3f (%s)' % (mean_cv_score_modified, cv_scores_modified))
    print('  Val acc (clean data)   : %.3f (%s)' % (mean_cv_score_train, cv_scores_train))
    print('')
    
    if mean_cv_score_modified > best_cv_score_modified:
        best_cv_score_modified = mean_cv_score_modified
        best_weight_decay_modified = weight_decay 
    if mean_cv_score_train > best_cv_score_train:
        best_cv_score_train = mean_cv_score_train
        best_weight_decay_train = weight_decay

print('Best value of weight_decay on modified data: %s' % best_weight_decay_modified)
print('Best value of weight_decay on clean data   : %s' % best_weight_decay_train)
print('\n')

results['weight_decays'] = weight_decays
results['best_weight_decay_modified'] = best_weight_decay_modified
results['best_weight_decay_train'] = best_weight_decay_train
results['cv_val_acc_modified'] = best_cv_score_modified
results['cv_val_acc_train'] = best_cv_score_train



### Evaluate on test set
print('## Evaluating on test set using cross-validated weight_decay, without any other defenses')

C = 1.0 / (X_train.shape[0] * best_weight_decay_train)        
svm_model = svm.LinearSVC(
    C=C,
    loss='hinge',
    tol=tol_svm,
    fit_intercept=True,
    random_state=random_state_svm,
    max_iter=max_iter_test_svm)

svm_model.fit(X_train, Y_train)
params_train = np.reshape(svm_model.coef_, -1)
bias_train = svm_model.intercept_[0]

results['test'] = defaultdict(dict)
results['test']['clean'] = defaultdict(dict)
results['test']['clean']['train_acc_clean'] = svm_model.score(X_train, Y_train)
results['test']['clean']['test_acc'] = svm_model.score(X_test, Y_test)
results['test']['clean']['params_norm'] = np.linalg.norm(params_train)

results['test']['clean']['train_loss_clean'] = hinge_loss(params_train, bias_train, X_train, Y_train)
results['test']['clean']['test_loss'] = hinge_loss(params_train, bias_train, X_test, Y_test)

print('Training on the clean data:')
print('  Train acc (clean) : ', results['test']['clean']['train_acc_clean'])
print('  Train loss (clean): ', results['test']['clean']['train_loss_clean'])
print('  Test acc          : ', results['test']['clean']['test_acc'])
print('  Test loss         : ', results['test']['clean']['test_loss'])
print('  Norm of params: ', results['test']['clean']['params_norm'])
print('')

C = 1.0 / (X_modified.shape[0] * best_weight_decay_modified)        
svm_model = svm.LinearSVC(
    C=C,
    loss='hinge',
    tol=tol_svm,
    fit_intercept=True,
    random_state=random_state_svm,
    max_iter=max_iter_test_svm)

svm_model.fit(X_modified, Y_modified)
params_modified = np.reshape(svm_model.coef_, -1)
bias_modified = svm_model.intercept_[0]

results['test']['modified'] = defaultdict(dict)
results['test']['modified']['train_acc_overall'] = svm_model.score(X_modified, Y_modified)
results['test']['modified']['train_acc_clean'] = svm_model.score(X_train, Y_train)
results['test']['modified']['train_acc_poison'] = svm_model.score(X_poison, Y_poison)
results['test']['modified']['test_acc'] = svm_model.score(X_test, Y_test)
results['test']['modified']['params_norm'] = np.linalg.norm(params_modified)

results['test']['modified']['train_loss_overall'] = hinge_loss(params_modified, bias_modified, X_modified, Y_modified)
results['test']['modified']['train_loss_clean'] = hinge_loss(params_modified, bias_modified, X_train, Y_train)
results['test']['modified']['train_loss_poison'] = hinge_loss(params_modified, bias_modified, X_poison, Y_poison)
results['test']['modified']['test_loss'] = hinge_loss(params_modified, bias_modified, X_test, Y_test)

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
leverage = diff_in_test_acc / datadef.epsilon
print('Difference in test accuracy between clean and poisoned: %.3f' % diff_in_test_acc)
print('Leverage: %.3f' % leverage)
print('\n')

results['test']['leverage'] = leverage

if (min_leverage is not None) and (leverage < min_leverage):
    print('Leverage is below specified min_leverage. Skipping defenses...')

elif no_defense:
    print('Skipping defenses...')

else:

    ### Q-based defenses
    print('## Running Q-based defenses')

    class_map = data.get_class_map()

    for use_emp, use_emp_label in [(True, 'emp'), (False, 'true')]:
        
        if use_emp:
            print('Using poisoned data to calculate centroids/covariances:')
            centroids = datadef.emp_centroids
        else:
            print('Using true data to calculate centroids/covariances:')
            centroids = datadef.true_centroids
        centroid_vec = data.get_centroid_vec(centroids)
            
        ## Distance in the direction of the vector between centroids ("slab")
        defense_label = 'centroid-vec'    
        if ((defense_to_test is None) or (defense_to_test == defense_label)):
            print('  Computing distances in the direction of the vector between centroids...')
            Q = centroid_vec
            norm = 2
            all_dists, results = defense_testers.process_defense(
                datadef, 
                Q=Q, 
                all_dists=all_dists, 
                model=svm_model, 
                results=results, 
                use_emp=use_emp, 
                use_emp_label=use_emp_label, 
                defense_label=defense_label,
                max_frac_to_remove=max_frac_to_remove, 
                frac_increment=frac_increment, 
                num_folds=num_folds)

        ## L2 distance to centroids ("L2 ball")
        defense_label = 'l2-ball'    
        if ((defense_to_test is None) or (defense_to_test == defense_label)):
            print('  Computing L2 distances to class centroids...')
            Q = None
            norm = 2
            all_dists, results = defense_testers.process_defense(
                datadef, 
                Q=Q, 
                all_dists=all_dists, 
                model=svm_model, 
                results=results, 
                use_emp=use_emp, 
                use_emp_label=use_emp_label, 
                defense_label=defense_label,
                max_frac_to_remove=max_frac_to_remove, 
                frac_increment=frac_increment, 
                num_folds=num_folds)

        ## L1 distance to centroids ("L1 ball")
        defense_label = 'l1-ball'    
        if ((defense_to_test is None) or (defense_to_test == defense_label)):
            print('  Computing L1 distances to class centroids...')
            Q = None
            norm = 1
            all_dists, results = defense_testers.process_defense(
                datadef, 
                Q=Q, 
                all_dists=all_dists, 
                model=svm_model, 
                results=results, 
                use_emp=use_emp, 
                use_emp_label=use_emp_label, 
                defense_label=defense_label,
                max_frac_to_remove=max_frac_to_remove, 
                frac_increment=frac_increment, 
                num_folds=num_folds,
                norm=norm)

        ## Inverse sigma 
        # If num_features > 2048, use top k PCs
        # Else just use the whole thing
        defense_label = 'inverse-sigma'
        if ((defense_to_test is None) or (defense_to_test == defense_label)):
            print('  Computing inverse sigma distances to class centroids...')
            if X_modified.shape[1] > 2048:
                P, sv_ratio, PX_modified, _, _ = datadef.project_to_low_rank(
                    k=inverse_sigma_dim,
                    use_emp=use_emp,
                    get_projected_data=True)
                P_datadef = defenses.DataDef(PX_modified, Y_modified, None, None, idx_train, idx_poison)
                print('    Projected to %s-d. sigma_k / sigma_1 = %.4f' % (inverse_sigma_dim, sv_ratio))
            else:
                P_datadef = datadef

            sqrt_inv_covs = P_datadef.get_sqrt_inv_covs(use_emp=use_emp)
            
            Q = sqrt_inv_covs
            all_dists, results = defense_testers.process_defense(
                datadef, 
                Q=Q, 
                all_dists=all_dists, 
                model=svm_model, 
                results=results, 
                use_emp=use_emp, 
                use_emp_label=use_emp_label, 
                defense_label=defense_label,
                max_frac_to_remove=max_frac_to_remove, 
                frac_increment=frac_increment, 
                num_folds=num_folds,
                P_datadef=P_datadef)

        ## PCA-subspace
        # How to pick k?
        defense_label = 'pca-subspace'
        if ((defense_to_test is None) or (defense_to_test == defense_label)):
            print('  Computing PCA subspace...')

            P, sv_ratio = datadef.project_to_low_rank(
                k=None,
                use_emp=use_emp,
                get_projected_data=False)
            print('    Normal subspace = first %s SVs + mu_pos + mu_neg. This measures distance in the abnormal subspace (everything outside the normal subspace).' % (P.shape[0] - 2))
            print('    sigma_k / sigma_1 = %.4f' % sv_ratio)
            results['pca_normal_k'] = P.shape[0]
            results['pca_normal_ratio'] = sv_ratio
            subtract_from_l2 = True
            all_dists, results = defense_testers.process_defense(
                datadef, 
                Q=P, 
                all_dists=all_dists, 
                model=svm_model, 
                results=results, 
                use_emp=use_emp, 
                use_emp_label=use_emp_label, 
                defense_label=defense_label,
                max_frac_to_remove=max_frac_to_remove, 
                frac_increment=frac_increment, 
                num_folds=num_folds,
                subtract_from_l2=subtract_from_l2)

    ## k-NN
    defense_label = 'k-NN'
    if ((defense_to_test is None) or (defense_to_test == defense_label)):
        print('  Computing k-nearest neighbors...')

        dists = datadef.get_knn_dists(num_neighbors=num_neighbors).reshape(-1, 1)
        all_dists, results = defense_testers.process_defense(
            datadef, 
            Q=None, 
            all_dists=all_dists, 
            model=svm_model, 
            results=results, 
            use_emp=None, 
            use_emp_label=None, 
            defense_label=defense_label,
            max_frac_to_remove=max_frac_to_remove, 
            frac_increment=frac_increment, 
            num_folds=num_folds,
            dists=dists)
        results['num_neighbors'] = num_neighbors

### Save dists and metadata
print('')
print('Defenses done. Saving output to disk...')
np.savez(
    output_dists_path, 
    all_dists=all_dists, 
    params_clean=params_train, 
    bias_clean=bias_train,
    params_poisoned=params_modified,
    bias_poisoned=bias_modified)

with open(output_json_path, 'w') as f:
    f.write(json.dumps(results, cls=data.NumpyEncoder))

sio.savemat(output_mat_path, {'params_clean':params_train, 'params_poisoned':params_modified})

### Execute notebook
print('')
print('Generating report...')
with open('viz_defenses.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

# Do surgery on the notebook to insert dataset and file names in the second cell
nb['cells'][1]['source'] = nb['cells'][1]['source'].replace('DATASET_NAME', dataset_name)
nb['cells'][1]['source'] = nb['cells'][1]['source'].replace('FILE_NAME', file_name)

ep = ExecutePreprocessor(timeout=600, kernel_name='python2')
ep.preprocess(nb, {})
with open(output_ipynb_path, 'wt') as f:
    nbformat.write(nb, f)

### Export completed notebook and copy it to web directory
html_exporter = HTMLExporter()
(body, resources) = html_exporter.from_file(output_ipynb_path)

with open(output_html_path, 'wt') as f:
    f.write(body)

shutil.copyfile(output_html_path, output_web_path)

print('All done! Check the report at %s.' % (output_html_path))

