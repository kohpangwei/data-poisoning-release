from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster, metrics, svm, model_selection, neighbors

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse

import data_utils as data


def remove_quantile(X, Y, dists, frac_to_remove):
    """
    Removes the frac_to_remove points from X and Y with the highest value in dists.
    This works separately for each class.
    """    
    if len(dists.shape) == 2: # Accept column vectors but reshape
        assert dists.shape[1] == 1
        dists = np.reshape(dists, -1)               
    
    assert len(dists.shape) == 1
    assert X.shape[0] == Y.shape[0]
    assert X.shape[0] == len(dists)
    assert 0 <= frac_to_remove
    assert frac_to_remove <= 1

    frac_to_keep = 1.0 - frac_to_remove

    idx_to_keep = []
    for y in set(Y): 
        num_to_keep = int(np.round(frac_to_keep * np.sum(Y == y)))

        idx_to_keep.append( 
            np.where(Y == y)[0][np.argsort(dists[Y == y])[:num_to_keep]])

    idx_to_keep = np.concatenate(idx_to_keep)

    X_def = X[idx_to_keep, :]
    Y_def = Y[idx_to_keep]

    return X_def, Y_def, idx_to_keep


def compute_dists_under_Q(    
    X, Y,
    Q,
    subtract_from_l2=False, #If this is true, computes ||x - mu|| - ||Q(x - mu)||
    centroids=None,
    class_map=None,    
    norm=2):
    """
    Computes ||Q(x - mu)|| in the corresponding norm. 
    Returns a vector of length num_examples (X.shape[0]).
    If centroids is not specified, calculate it from the data.
    If Q has dimension 3, then each class gets its own Q.
    """
    if (centroids is not None) or (class_map is not None): 
        assert (centroids is not None) and (class_map is not None)
    if subtract_from_l2: 
        assert Q is not None
    if Q is not None and len(Q.shape) == 3:
        assert class_map is not None
        assert Q.shape[0] == len(class_map)

    if norm == 1:
        metric = 'manhattan'
    elif norm == 2:
        metric = 'euclidean'
    else:
        raise ValueError, 'norm must be 1 or 2'

    Q_dists = np.zeros(X.shape[0])
    if subtract_from_l2:
        L2_dists = np.zeros(X.shape[0])    

    for y in set(Y): 
        if centroids is not None:
            mu = centroids[class_map[y], :]
        else:
            mu = np.mean(X[Y == y, :], axis=0)
        mu = mu.reshape(1, -1)

        if Q is None:   # assume Q = identity  
            Q_dists[Y == y] = metrics.pairwise.pairwise_distances(
                X[Y == y, :],
                mu,
                metric=metric)            
        else:
            if len(Q.shape) == 3:
                current_Q = Q[class_map[y], ...]
            else:
                current_Q = Q

            if sparse.issparse(X):
                XQ = X[Y == y, :].dot(current_Q.T)
            else:
                XQ = current_Q.dot(X[Y == y, :].T).T
            muQ = current_Q.dot(mu.T).T

            Q_dists[Y == y] = metrics.pairwise.pairwise_distances(
                XQ,
                muQ,
                metric=metric)   

            if subtract_from_l2:
                L2_dists[Y == y] = metrics.pairwise.pairwise_distances(
                    X[Y == y, :],
                    mu,
                    metric=metric) 
                Q_dists[Y == y] = np.sqrt(np.square(L2_dists[Y == y]) - np.square(Q_dists[Y == y]))

    return Q_dists



class DataDef(object):
    def __init__(self, X_modified, Y_modified, X_test, Y_test, idx_train, idx_poison):
        self.X_modified = X_modified
        self.Y_modified = Y_modified
        self.X_test = X_test
        self.Y_test = Y_test
        self.idx_train = idx_train
        self.idx_poison = idx_poison

        self.X_train = X_modified[idx_train, :]
        self.Y_train = Y_modified[idx_train]
        self.X_poison = X_modified[idx_poison, :]
        self.Y_poison = Y_modified[idx_poison]

        self.class_map = data.get_class_map()
        self.emp_centroids = data.get_centroids(self.X_modified, self.Y_modified, self.class_map)
        self.true_centroids = data.get_centroids(self.X_train, self.Y_train, self.class_map)
        self.emp_centroid_vec = data.get_centroid_vec(self.emp_centroids)
        self.true_centroid_vec = data.get_centroid_vec(self.true_centroids)

        # Fraction of bad data / good data (so in total, there's 1+epsilon * good data )
        self.epsilon = self.X_poison.shape[0] / self.X_train.shape[0]

    def plot_dists(
        self,
        dists, 
        num_bins=100):    

        dists_train = dists[self.idx_train]
        dists_poison = dists[self.idx_poison]

        for y in set(self.Y_modified): 
            if np.sum(self.Y_poison == y) == 0:
                print('No poisoned examples were from class %s.' % y)
            else:
                lower = np.min((np.min(dists_train), np.min(dists_poison))) * 0.9
                upper = np.max((np.max(dists_train), np.max(dists_poison))) * 1.1
                    
                step = (upper - lower) / num_bins
                bins=np.arange(lower, upper, step)
                plt.figure(figsize=(10, 5))
                plt.title('y = %s (# clean points = %s, # poisoned points = %s)' % (y, np.sum(self.Y_train == y), np.sum(self.Y_poison == y)))
                sns.distplot(dists_train[self.Y_train == y], kde=False, bins=bins)
                if np.sum(self.Y_poison == y) == 1:
                    # Hack to make this a "distribution"
                    temp_dists = np.concatenate((dists_poison[self.Y_poison == y], dists_poison[self.Y_poison == y]))
                    sns.distplot(temp_dists, kde=False, bins=bins, rug=True)
                else:
                    sns.distplot(dists_poison[self.Y_poison == y], kde=False, bins=bins)
                plt.show()


    def compute_dists_under_Q_over_dataset(
        self,
        Q,
        subtract_from_l2=False, #If this is true, plots ||x - mu|| - ||Q(x - mu)||
        use_emp_centroids=False,        
        norm=2):

        if use_emp_centroids:
            centroids = self.emp_centroids
        else:
            centroids = self.true_centroids
        
        dists = compute_dists_under_Q(    
            self.X_modified, self.Y_modified,
            Q,
            subtract_from_l2=subtract_from_l2,
            centroids=centroids,
            class_map=self.class_map,    
            norm=norm)        

        return dists


    def get_sqrt_inv_covs(self, use_emp=False):
        if use_emp:            
            sqrt_inv_covs = data.get_sqrt_inv_cov(self.X_modified, self.Y_modified, self.class_map)
        else:            
            sqrt_inv_covs = data.get_sqrt_inv_cov(self.X_train, self.Y_train, self.class_map)
        return sqrt_inv_covs


    def get_knn_dists(self, num_neighbors):

        nbrs = neighbors.NearestNeighbors(
            n_neighbors=num_neighbors, 
            metric='euclidean').fit(
                self.X_modified)
        dists_to_each_neighbor, _ = nbrs.kneighbors(self.X_modified)
        return np.sum(dists_to_each_neighbor, axis=1)


    # Might be able to speed up; is svds actually performant on dense matrices?
    def project_to_low_rank(
        self,
        k,
        use_emp=False,
        get_projected_data=False):
        """
        Projects to the rank (k+2) subspace defined by the top k SVs, mu_pos, and mu_neg.

        If k is None, it tries to find a good k by taking the top 1000 SVs and seeing if we can
        find some k such that sigma_k / sigma_1 < 0.1. If we can, we take the smallest such k. 
        If not, we take k = 1000. 
        """
        if use_emp:
            X = self.X_modified
            Y = self.Y_modified
        else:
            X = self.X_train
            Y = self.Y_train
        
        if k is not None:
            assert k > 0
            assert k < self.X_train.shape[1]

            U, S, V = sparse.linalg.svds(X, k=k, which='LM') 
        
        # If k is not specified, try to automatically find a good value
        else:
            search_k = 1000
            target_sv_ratio = 0.01

            U, S, V = sparse.linalg.svds(X, k=search_k, which='LM') 
            # Make sure it's sorted in the order we think it is...
            max_sv = np.max(S)
            assert S[-1] == max_sv
            sv_ratios = S / S[-1] 
            if np.min(sv_ratios) < target_sv_ratio:
                k = np.where(sv_ratios > target_sv_ratio)[0][0]
            else:
                print('  Giving up -- min ratio was %s' % np.min(sv_ratios))
                k = 1                
            V = V[k-1:, :]

        mu_pos = np.array(np.mean(X[Y == 1, :], axis=0)).reshape(1, -1)
        mu_neg = np.array(np.mean(X[Y == -1, :], axis=0)).reshape(1, -1)
            
        V_mu = np.concatenate((V, mu_pos, mu_neg), axis=0)
        P = slin.orth(V_mu.T).T
        
        achieved_sv_ratio = S[k-1] / S[-1]

        if get_projected_data:
            PX_modified = self.X_modified.dot(P.T)
            PX_train = self.X_train.dot(P.T)
            PX_poison = self.X_poison.dot(P.T)
            return P, achieved_sv_ratio, PX_modified, PX_train, PX_poison
        else:
            return P, achieved_sv_ratio


    def find_num_points_kept(self, idx_to_keep):
        good_mask = np.zeros(self.X_modified.shape[0], dtype=bool)
        good_mask[self.idx_train] = True
        bad_mask = np.zeros(self.X_modified.shape[0], dtype=bool)
        bad_mask[self.idx_poison] = True

        keep_mask = np.zeros(self.X_modified.shape[0], dtype=bool)
        keep_mask[idx_to_keep] = True
    
        frac_of_good_points_kept = np.mean(keep_mask & good_mask) / np.mean(good_mask)
        frac_of_bad_points_kept = np.mean(keep_mask & bad_mask) / np.mean(bad_mask)
        return frac_of_good_points_kept, frac_of_bad_points_kept


    def remove_and_retrain(
        self,
        dists,
        model,
        frac_to_remove,
        num_folds=5):

        X_def, Y_def, idx_to_keep = remove_quantile(
            self.X_modified, 
            self.Y_modified, 
            dists=dists, 
            frac_to_remove=frac_to_remove)

        frac_of_good_points_kept, frac_of_bad_points_kept = self.find_num_points_kept(idx_to_keep)

        k_fold = model_selection.KFold(n_splits=num_folds, shuffle=True, random_state=2)

        cv_scores = model_selection.cross_val_score(
            model, 
            X_def, Y_def, 
            cv=k_fold, 
            n_jobs=np.min((num_folds, 8)))        
        mean_cv_score = np.mean(cv_scores)

        model.fit(X_def, Y_def)
        train_acc = model.score(X_def, Y_def)
        test_acc = model.score(self.X_test, self.Y_test)

        return train_acc, mean_cv_score, test_acc, frac_of_good_points_kept, frac_of_bad_points_kept
