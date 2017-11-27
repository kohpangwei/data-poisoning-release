# Go through each quartile and copy the influence_data folder

import os 
from shutil import copyfile

os.makedirs('influence_data')

for qdir in ['q0', 'q1', 'q2', 'q3']:
     for filename in os.listdir(os.path.join(qdir, 'influence_data')):
        if filename[-4:] != '.npz': continue        
        src = os.path.join(qdir, 'influence_data', filename)
        dst = os.path.join('influence_data', filename)
        copyfile(src, dst)

        