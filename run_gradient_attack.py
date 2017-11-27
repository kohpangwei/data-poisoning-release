from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import os
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse
from scipy.stats import pearsonr

import datasets
import data_utils as data

import tensorflow as tf
from influence.influence.smooth_hinge import SmoothHinge

from influence.influence.dataset import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
import iterative_attack


parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', help='One of: imdb, enron, dogfish, mnist_17')
parser.add_argument('--shard', type=int)

args = parser.parse_args()
dataset_name = args.dataset_name
shard = args.shard

norm_sq_constraint = datasets.DATASET_NORM_SQ_CONSTRAINTS[dataset_name]

X_train, Y_train, X_test, Y_test = datasets.load_dataset(dataset_name)
if sparse.issparse(X_train):
    X_train = X_train.toarray()
if sparse.issparse(X_test):
    X_test = X_test.toarray()

train = DataSet(X_train, Y_train)
validation = None
# We want to directly attack the clean train data
# so we pretend that it's the test data
test = DataSet(X_train, Y_train)
data_sets = base.Datasets(train=train, validation=validation, test=test)

temp = 0
input_dim = X_train.shape[1]
weight_decay = 0.01

if X_train.shape[0] % 100 == 0:
    batch_size = 100
else:
    batch_size = X_train.shape[0]
initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]
num_classes = 2

step_size = 0.001
# step_size = 0.01

project_sphere = True
project_slab = True
label_flip = True

output_root = os.path.join(datasets.OUTPUT_FOLDER, 'influence_data')

standard_bounds_path = datasets.get_bounds_path(dataset_name, norm_sq_constraint)

standard_f = np.load(standard_bounds_path)

epsilons = standard_f['epsilons']
lower_weight_decays = standard_f['lower_weight_decays']

assert np.all(datasets.DATASET_EPSILONS[dataset_name] == epsilons)
assert epsilons[0] == 0
assert len(epsilons) == 9
if shard == 0:
    target_epsilons = epsilons[1:3]
elif shard == 1:
    target_epsilons = epsilons[3:5]
elif shard == 2:
    target_epsilons = epsilons[5:6]
elif shard == 3:
    target_epsilons = epsilons[6:7]
elif shard == 4:
    target_epsilons = epsilons[7:8]
elif shard == 5:
    target_epsilons = epsilons[8:9]
else:
    raise ValueError, 'shard must be 0-5'    

for random_seed in [1]:
    for epsilon_idx, epsilon in enumerate(epsilons):
        print('========== Epsilon %s ==========' % epsilon)
        if epsilon not in target_epsilons:
            continue
        
        total_copies = int(np.round(epsilon * X_train.shape[0]))
        num_pos_copies = int(total_copies / 2)
        num_neg_copies = total_copies - num_pos_copies

        weight_decay = lower_weight_decays[epsilon_idx]

        tf.reset_default_graph()

        model = SmoothHinge(            
            input_dim=input_dim,
            temp=temp,
            weight_decay=weight_decay,
            use_bias=True,
            num_classes=num_classes,
            batch_size=batch_size,
            data_sets=data_sets,
            initial_learning_rate=initial_learning_rate,
            decay_epochs=None,
            mini_batch=False,
            train_dir=output_root,
            log_dir='log',
            model_name='smooth_hinge_%s_sphere-%s_slab-%s_start-copy_lflip-%s_step-%s_t-%s_eps-%s_wd-%s_rs-%s' % (
                dataset_name, project_sphere, project_slab, label_flip, 
                step_size, temp, epsilon, weight_decay, random_seed))

        X_modified, Y_modified = data.copy_random_points(
            X_train, Y_train, 
            target_class=1, 
            num_copies=num_pos_copies, 
            random_seed=random_seed, 
            replace=True)
        X_modified, Y_modified = data.copy_random_points(
            X_modified, Y_modified, 
            target_class=-1, 
            num_copies=num_neg_copies, 
            random_seed=random_seed, 
            replace=True)

        if label_flip:
            Y_modified[X_train.shape[0]:] = -Y_modified[X_train.shape[0]:]

        model.update_train_x_y(X_modified, Y_modified)
        model.train()

        projection_fn = data.get_projection_fn(
            X_train, Y_train,
            sphere=project_sphere,
            slab=project_slab,
            percentile=70)

        iterative_attack.iterative_attack(
            model, 
            indices_to_poison=np.arange(X_train.shape[0], X_modified.shape[0]),            
            test_idx=None, 
            test_description=None, 
            step_size=step_size, 
            num_iter=2000,
            loss_type='normal_loss',
            projection_fn=projection_fn,
            output_root=output_root)