# Go through each shard and copy the influence_data folder

import os 
from shutil import copyfile

os.makedirs('influence_data')

for sdir in ['s0', 's1', 's2', 's3', 's4', 's5']:
     for filename in os.listdir(os.path.join(sdir, 'influence_data')):
        if filename[-4:] != '.npz': continue        
        src = os.path.join(sdir, 'influence_data', filename)
        dst = os.path.join('influence_data', filename)
        copyfile(src, dst)
