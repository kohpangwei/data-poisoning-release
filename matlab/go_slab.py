from subprocess import call, Popen
import shlex

import sys

import datasets

procs = []

quantiles = {'mnist_17': 0.65, 'dogfish': 0.50}

for dataset_name in ['mnist_17', 'dogfish']:

    rho_squared = datasets.DATASET_NORM_SQ_CONSTRAINTS[dataset_name]
    quantile = quantiles[dataset_name]

    for epsilon in datasets.DATASET_EPSILONS[dataset_name]:
        if epsilon == 0: continue

        eta = 0.005 

        log_file = '%s_slab_eps%s_rhosq%s_quantile%s_v7.log' % (dataset_name, epsilon, rho_squared, quantile)
        err_file = '%s_slab_eps%s_rhosq%s_quantile%s_v7.log' % (dataset_name, epsilon, rho_squared, quantile)

        cmd =  './matlab/bin/matlab -r "slabAttack(\'%s\',%s,%s,%s,%s,\'sedumi\',200);exit"' % (
                dataset_name,
                epsilon,
                eta,
                rho_squared,
                quantile)
        print('running command: %s' % cmd)
        proc = Popen(shlex.split(cmd), stdout=open(log_file,'w'), stderr=open(err_file,'w'))
        procs.append(proc)

for proc in procs:
  proc.wait()
  print('done with command %s' % proc.pid)
