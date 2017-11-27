from subprocess import call

import sys

import datasets

for dataset_name in ['enron']:

    rho_squared = datasets.DATASET_NORM_SQ_CONSTRAINTS[dataset_name]
    quantile = 0.70

    for epsilon in datasets.DATASET_EPSILONS[dataset_name]:
        if epsilon == 0: continue

        eta = (0.008 / epsilon) * (epsilon / 0.05) ** (0.5)

        log_file = '%s_integer_eps%s_rhosq%s_IQP_v2.log' % (dataset_name, epsilon, rho_squared)

        call(
            './matlab/bin/matlab -r "feasibilityAttack(\'%s\',%s,%s,%s,%s,\'gurobi\');exit" > %s &' % (
                dataset_name,
                epsilon,
                eta,
                rho_squared,
                quantile,
                log_file),
            shell=True)


