# Go through each slab bundle and copy mat files into the slab_mat folder
# Rename files to fit datasets.py code

import os 
from shutil import copyfile

os.makedirs('slab_mat')

for filename in os.listdir('input'):
    if filename[-4:] != '.mat': continue 
    src = os.path.join('input', filename)
    dst = os.path.join('slab_mat', filename.replace('quantile65_rho_slab_v7.mat', 'rho_slab_v6.mat'))
    copyfile(src, dst)

