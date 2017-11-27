from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import os 
import numpy as np
import scipy.sparse as sparse
import scipy.io as sio
import IPython

# Local running
DATA_FOLDER = '/afs/cs.stanford.edu/u/jsteinhardt/data-poisoning-scratch/data'
OUTPUT_FOLDER = '/afs/cs.stanford.edu/u/jsteinhardt/data-poisoning-scratch'

# Codalab
# DATA_FOLDER = './data'
# OUTPUT_FOLDER = '.'

# WEB_FOLDER = '/afs/cs.stanford.edu/u/jsteinhardt/www/reports'

DATASET_NORM_SQ_CONSTRAINTS = {
    'imdb': 16, 
    'enron': 6, 
    'dogfish': 0.5, 
    'mnist_17': 0.8    
}   

small_epsilons = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
large_epsilons = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
DATASET_EPSILONS = {
    'imdb': small_epsilons,
    'enron': small_epsilons,
    'dogfish': large_epsilons,
    'mnist_17': large_epsilons
}

DATASET_NUM_ITERS_AFTER_BURNIN = {
    'imdb': 6000,
    'enron': 8000,
    'dogfish': 15000,
    'mnist_17': 8000
}

DATASET_LEARNING_RATES = {
    'imdb': 0.001,
    'enron': 0.1,
    'dogfish': 0.05, 
    'mnist_17': 0.1
}

PERCENTILE = 70

def safe_makedirs(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: 
            if exc.errno != errno.EEXIST:
                raise

def get_baseline_npz_path(dataset_name, baseline_name):
    return os.path.join(
        OUTPUT_FOLDER, 
        dataset_name, 
        '%s_baseline-%s_results.npz' % (baseline_name, dataset_name))


# def get_web_html_path(file_name):
    # return os.path.join(WEB_FOLDER, '%s_defense_report.html' % file_name)


def get_output_mat_path(dataset_name, file_name):
    return os.path.join(
        OUTPUT_FOLDER,
        dataset_name,
        '%s_defense_params.mat' % file_name)


def get_output_dists_path(dataset_name, file_name):
    return os.path.join(
        OUTPUT_FOLDER,
        dataset_name,
        '%s_defense_dists.npz' % file_name)


def get_output_json_path(dataset_name, file_name):
    return os.path.join(
        OUTPUT_FOLDER,
        dataset_name,
        '%s_defense_results.json' % file_name)


def get_output_ipynb_path(dataset_name, file_name):
    return os.path.join(
        OUTPUT_FOLDER,
        dataset_name,
        '%s_defense_report.ipynb' % file_name)


def get_output_html_path(dataset_name, file_name):
    return os.path.join(
        OUTPUT_FOLDER,
        dataset_name,
        '%s_defense_report.html' % file_name)


def get_bounds_path(dataset_name, norm_sq_constraint, percentile=PERCENTILE):
    if percentile == PERCENTILE:
        return os.path.join(OUTPUT_FOLDER, 'bounds', '%s_normc-%s_bounds.npz' % (dataset_name, norm_sq_constraint))
    else:
        return os.path.join(OUTPUT_FOLDER, 'bounds', '%s_normc-%s_percentile-%s_bounds.npz' % (dataset_name, norm_sq_constraint, percentile))


def get_attack_npz_filename(dataset_name, epsilon, norm_sq_constraint, percentile=PERCENTILE):
    if percentile == PERCENTILE:
        return '%s_attack_clean-centroid_normc-%s_epsilon-%s.npz' % (dataset_name, norm_sq_constraint, epsilon)    
    else:
        return '%s_attack_clean-centroid_normc-%s_percentile-%s_epsilon-%s.npz' % (dataset_name, norm_sq_constraint, percentile, epsilon)    


def get_attack_npz_path(dataset_name, epsilon, norm_sq_constraint, percentile=PERCENTILE):
    return os.path.join(OUTPUT_FOLDER, 'attack', get_attack_npz_filename(dataset_name, epsilon, norm_sq_constraint, percentile))


### Integer / feasible
def get_int_bounds_path(dataset_name, norm_sq_constraint):
    return os.path.join(OUTPUT_FOLDER, 'int_bounds', '%s_int_normc-%s_bounds.npz' % (dataset_name, norm_sq_constraint))


def get_int_attack_npz_filename(dataset_name, epsilon, norm_sq_constraint, percentile=PERCENTILE):
    if percentile == PERCENTILE:
        return '%s_attack_clean-centroid_int_normc-%s_epsilon-%s.npz' % (dataset_name, norm_sq_constraint, epsilon)
    else:
        return '%s_attack_clean-centroid_int_normc-%s_percentile-%s_epsilon-%s.npz' % (dataset_name, norm_sq_constraint, percentile, epsilon)


def get_int_attack_npz_path(dataset_name, epsilon, norm_sq_constraint, percentile=PERCENTILE):
    return os.path.join(OUTPUT_FOLDER, 'attack', get_int_attack_npz_filename(dataset_name, epsilon, norm_sq_constraint, percentile))


def get_int_mat_path(dataset_name, epsilon):
    return os.path.join(OUTPUT_FOLDER, 'int_mat', '%s_attack_eps%s_rho_integer_IQP_v2.mat' % (dataset_name, str('{:.2f}'.format(epsilon))[-2:]))
    

### Slab
def get_slab_bounds_path(dataset_name,  norm_sq_constraint):
    return os.path.join(OUTPUT_FOLDER, 'slab_bounds', '%s_slab_normc-%s_bounds.npz' % (dataset_name, norm_sq_constraint))


def get_slab_attack_npz_filename(dataset_name, epsilon, norm_sq_constraint):
    return '%s_attack_clean-centroid_slab_normc-%s_epsilon-%s.npz' % (dataset_name, norm_sq_constraint, epsilon)


def get_slab_attack_npz_path(dataset_name, epsilon, norm_sq_constraint):
    return os.path.join(OUTPUT_FOLDER, 'attack', get_slab_attack_npz_filename(dataset_name, epsilon, norm_sq_constraint))


def get_slab_mat_path(dataset_name, epsilon, percentile=None):
    # return os.path.join(OUTPUT_FOLDER, 'slab_mat', '%s_attack_eps%s_rho_slab_v3.mat' % (dataset_name, str('{:.2f}'.format(epsilon))[-2:]))  
    if percentile:
        return os.path.join(OUTPUT_FOLDER, 'slab_mat', '%s_attack_eps%s_quantile%s_rho_slab_v7.mat' % (dataset_name, str('{:.2f}'.format(epsilon))[-2:], percentile))  
    else:
        return os.path.join(OUTPUT_FOLDER, 'slab_mat', '%s_attack_eps%s_rho_slab_v6.mat' % (dataset_name, str('{:.2f}'.format(epsilon))[-2:]))  


### Grad
def get_grad_bounds_path(dataset_name, norm_sq_constraint):
    return os.path.join(OUTPUT_FOLDER, 'grad_bounds', '%s_grad_normc-%s_bounds.npz' % (dataset_name, norm_sq_constraint))

def get_grad_attack_wd_npz_filename(dataset_name, epsilon, weight_decay):    
    return 'smooth_hinge_%s_sphere-True_slab-True_start-copy_lflip-True_step-0.001_t-0_eps-%s_wd-%s_rs-1_attack.npz' % (dataset_name, epsilon, weight_decay)

def get_grad_attack_wd_npz_path(dataset_name, epsilon, weight_decay):
    return os.path.join(OUTPUT_FOLDER, 'influence_data', get_grad_attack_wd_npz_filename(dataset_name, epsilon, weight_decay))

def get_grad_attack_npz_filename(dataset_name, epsilon, norm_sq_constraint):
    return '%s_attack_clean-centroid_grad_normc-%s_epsilon-%s.npz' % (dataset_name, norm_sq_constraint, epsilon)

def get_grad_attack_npz_path(dataset_name, epsilon, norm_sq_constraint):
    return os.path.join(OUTPUT_FOLDER, 'attack', get_grad_attack_npz_filename(dataset_name, epsilon, norm_sq_constraint))

### Labelflip
# Doesn't actually use norm_sq_constraint, but takes it in for interface
def get_labelflip_bounds_path(dataset_name, norm_sq_constraint):
    return os.path.join(OUTPUT_FOLDER, 'labelflip_bounds', '%s_labelflip_bounds.npz' % (dataset_name))


def get_labelflip_attack_npz_filename(dataset_name, epsilon, norm_sq_constraint):
    return '%s_attack_clean-centroid_labelflip_epsilon-%s.npz' % (dataset_name, epsilon)


def get_labelflip_attack_npz_path(dataset_name, epsilon, norm_sq_constraint):
    return os.path.join(OUTPUT_FOLDER, 'attack', get_labelflip_attack_npz_filename(dataset_name, epsilon, norm_sq_constraint))



def check_orig_data(X_train, Y_train, X_test, Y_test):
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_test.shape[0] == Y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert np.max(Y_train) == 1, 'max of Y_train was %s' % np.max(Y_train)
    assert np.min(Y_train) == -1
    assert len(set(Y_train)) == 2
    assert set(Y_train) == set(Y_test)
    

def check_poisoned_data(X_train, Y_train, X_poison, Y_poison, X_modified, Y_modified):
    assert X_train.shape[1] == X_poison.shape[1]
    assert X_train.shape[1] == X_modified.shape[1]
    assert X_train.shape[0] + X_poison.shape[0] == X_modified.shape[0]
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_poison.shape[0] == Y_poison.shape[0]
    assert X_modified.shape[0] == Y_modified.shape[0]
    assert X_train.shape[0] * X_poison.shape[0] * X_modified.shape[0] > 0
    

def load_dogfish():
    dataset_path = os.path.join(DATA_FOLDER)

    train_f = np.load(os.path.join(dataset_path, 'dogfish_900_300_inception_features_train.npz'))
    X_train = train_f['inception_features_val']
    Y_train = np.array(train_f['labels'] * 2 - 1, dtype=int)

    test_f = np.load(os.path.join(dataset_path, 'dogfish_900_300_inception_features_test.npz'))
    X_test = test_f['inception_features_val'] 
    Y_test = np.array(test_f['labels'] * 2 - 1, dtype=int)

    check_orig_data(X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, X_test, Y_test


def load_enron_sparse():
    dataset_path = os.path.join(DATA_FOLDER)
    f = np.load(os.path.join(dataset_path, 'enron1_processed_sparse.npz'))

    X_train = f['X_train'].reshape(1)[0]
    Y_train = f['Y_train'] * 2 - 1
    X_test = f['X_test'].reshape(1)[0]
    Y_test = f['Y_test'] * 2 - 1

    assert(sparse.issparse(X_train))
    assert(sparse.issparse(X_test))    

    check_orig_data(X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, X_test, Y_test


def load_imdb_sparse():
    dataset_path = os.path.join(DATA_FOLDER)
    f = np.load(os.path.join(dataset_path, 'imdb_processed_sparse.npz'))

    X_train = f['X_train'].reshape(1)[0]
    Y_train = f['Y_train'].reshape(-1)
    X_test = f['X_test'].reshape(1)[0]
    Y_test = f['Y_test'].reshape(-1)

    assert(sparse.issparse(X_train))
    assert(sparse.issparse(X_test))    

    check_orig_data(X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, X_test, Y_test


def load_dataset(dataset_name):
    if dataset_name == 'imdb':
        return load_imdb_sparse()
    elif dataset_name == 'enron':
        return load_enron_sparse()
    elif dataset_name == 'dogfish':
        return load_dogfish()
    else:
        dataset_path = os.path.join(DATA_FOLDER)
        f = np.load(os.path.join(dataset_path, '%s_train_test.npz' % dataset_name))

        X_train = f['X_train']
        Y_train = f['Y_train'].reshape(-1)
        X_test = f['X_test']
        Y_test = f['Y_test'].reshape(-1)

        check_orig_data(X_train, Y_train, X_test, Y_test)
        return X_train, Y_train, X_test, Y_test


def load_mnist_binary():
    return load_dataset('mnist_binary')


def load_attack(dataset_name, file_name):
    file_root, ext = os.path.splitext(file_name)

    if ext == '.mat':
        return load_attack_mat(dataset_name, file_name)
    elif ext == '.npz':
        return load_attack_npz(dataset_name, file_name)    
    else:
        raise ValueError, 'File extension must be .mat or .npz.'


def load_attack_mat(dataset_name, file_name, take_path=False):
    if take_path:
        file_path = file_name
    else:
        file_path = os.path.join(OUTPUT_FOLDER, 'mat', file_name)
    f = sio.loadmat(file_path)

    X_train = f['X_train'] 
    Y_train = f['y_train'].reshape(-1)
    if 'X_pert' in f:        
        X_poison = f['X_pert']
        Y_poison = f['y_pert'].reshape(-1)
    else:        
        X_poison = f['bestX'][0, ...]
        Y_poison = f['bestY'][0, ...].reshape(-1)
            
    X_test = f['X_test']
    Y_test = f['y_test'].reshape(-1)

    if not sparse.issparse(X_train):
        if sparse.issparse(X_poison):
            print('Warning: X_train is not sparse but X_poison is sparse. Densifying X_poison...')
            X_poison = X_poison.toarray()

    for X in [X_train, X_poison, X_test]:
        if sparse.issparse(X): X = X.tocsr()

    if sparse.issparse(X_train):
        X_modified = sparse.vstack((X_train, X_poison), format='csr')
    else:
        X_modified = np.concatenate((X_train, X_poison), axis=0)

    Y_modified = np.concatenate((Y_train, Y_poison), axis=0)

    # Create views into X_modified so that we don't have to keep copies lying around
    num_train = np.shape(X_train)[0]
    idx_train = slice(0, num_train)
    idx_poison = slice(num_train, np.shape(X_modified)[0])
    X_train = X_modified[idx_train, :]
    Y_train = Y_modified[idx_train]
    X_poison = X_modified[idx_poison, :]
    Y_poison = Y_modified[idx_poison]

    check_orig_data(X_train, Y_train, X_test, Y_test)
    check_poisoned_data(X_train, Y_train, X_poison, Y_poison, X_modified, Y_modified)

    return X_modified, Y_modified, X_test, Y_test, idx_train, idx_poison


def load_attack_npz(dataset_name, file_name, take_path=False):
    if take_path:
        file_path = file_name
    else:
        file_path = os.path.join(OUTPUT_FOLDER, 'attack', file_name)

    f = np.load(file_path)    

    if 'X_modified' in f:
        X_modified = f['X_modified']    
        Y_modified = f['Y_modified']
        X_test = f['X_test']
        Y_test = f['Y_test']
        idx_train = f['idx_train'].reshape(1)[0]
        idx_poison = f['idx_poison'].reshape(1)[0]
        # Extract sparse array from array wrapper
        if dataset_name in ['enron', 'imdb']:
            X_modified = X_modified.reshape(1)[0]
            X_test = X_test.reshape(1)[0]        
        
        X_train = X_modified[idx_train, :]
        Y_train = Y_modified[idx_train]
        X_poison = X_modified[idx_poison, :]
        Y_poison = Y_modified[idx_poison] 

    # This is for loading the baselines
    else:
        X_modified = f['poisoned_X_train']
        if dataset_name in ['enron', 'imdb']:
            try:
                X_modified = X_modified.reshape(1)[0]
            except:
                pass

        Y_modified = f['Y_train']

        X_train, Y_train, X_test, Y_test = load_dataset(dataset_name)
        
        idx_train = slice(0, X_train.shape[0])
        idx_poison = slice(X_train.shape[0], X_modified.shape[0])
        
        if sparse.issparse(X_modified):
            assert((X_modified[idx_train, :] - X_train).nnz == 0)
        else:
            if sparse.issparse(X_train):
                X_train = X_train.toarray()
                X_test = X_test.toarray()
            assert(np.all(np.isclose(X_modified[idx_train, :], X_train)))
        assert(np.all(Y_modified[idx_train] == Y_train))
        X_poison = X_modified[idx_poison, :]
        Y_poison = Y_modified[idx_poison]        

    check_orig_data(X_train, Y_train, X_test, Y_test)
    check_poisoned_data(X_train, Y_train, X_poison, Y_poison, X_modified, Y_modified)

    return X_modified, Y_modified, X_test, Y_test, idx_train, idx_poison


def extract_clean_data_from_mat(dataset_name, mat_name):
    mat_path = os.path.join(DATA_FOLDER, mat_name)

    f = sio.loadmat(mat_path)

    X_train = f['X_train'] 
    Y_train = f['y_train'].reshape(-1)    
    X_test = f['X_test']
    Y_test = f['y_test'].reshape(-1) 

    check_orig_data(X_train, Y_train, X_test, Y_test)

    output_npz_path = os.path.join(DATA_FOLDER, '%s_train_test.npz' % dataset_name)
    np.savez(
        output_npz_path,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test)

    print('Clean data saved to %s' % output_npz_path)


def process_matlab_train_test(
    f, 
    X_train, Y_train,
    X_test, Y_test,
    X_poison, Y_poison):

    if sparse.issparse(f['X_train']):
        assert np.all(f['X_train'].toarray() == X_train)
        assert np.all(f['X_test'].toarray() == X_test)
        X_modified = sparse.vstack((X_train, X_poison), format='csr')
    else:
        assert np.all(f['X_train'] == X_train)
        assert np.all(f['X_test'] == X_test)    
        X_modified = np.concatenate((X_train, X_poison), axis=0)

    Y_modified = np.concatenate((Y_train, Y_poison), axis=0)

    assert np.all(f['y_train'].reshape(-1) == Y_train)        
    assert np.all(f['y_test'].reshape(-1) == Y_test)

    idx_train = slice(0, X_train.shape[0])
    idx_poison = slice(X_train.shape[0], X_modified.shape[0])            

    check_orig_data(X_train, Y_train, X_test, Y_test)
    check_poisoned_data(X_train, Y_train, X_poison, Y_poison, X_modified, Y_modified)

    return X_modified, Y_modified, idx_train, idx_poison