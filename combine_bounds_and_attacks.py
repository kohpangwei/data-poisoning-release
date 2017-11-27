# Combine bounds and attack folders
# Reading from:
# mnist_normal_bounds
# mnist_grad_bounds
# mnist_slab_bounds
# dogfish_normal_bounds
# dogfish_slab_bounds
# imdb_normal_bounds
# enron_normal_bounds

import os 
from shutil import copyfile

os.makedirs('bounds')
os.makedirs('int_bounds')
os.makedirs('grad_bounds')
os.makedirs('labelflip_bounds')
os.makedirs('slab_bounds')
os.makedirs('attack')

for dirname in [
    'mnist_normal_bounds',
    'mnist_grad_bounds',
    'mnist_slab_bounds',
    'dogfish_normal_bounds',
    'dogfish_slab_bounds',
    'enron_normal_bounds',
    'enron_int_bounds',
    'imdb_normal_bounds',
]:
    for bound in ['bounds', 'int_bounds', 'slab_bounds', 'grad_bounds', 'labelflip_bounds']:
        if os.path.exists(os.path.join(dirname, bound)):
            for filename in os.listdir(os.path.join(dirname, bound)):
                if filename[-4:] != '.npz': continue 
                src = os.path.join(dirname, bound, filename)
                dst = os.path.join(bound, filename)
                copyfile(src, dst)

src = 'mnist_normal_bounds/attack/mnist_17_attack_clean-centroid_normc-0.8_epsilon-0.3.npz'
dst = 'attack/mnist_17_attack_clean-centroid_normc-0.8_epsilon-0.3.npz'
copyfile(src, dst)

src = 'imdb_normal_bounds/attack/imdb_attack_clean-centroid_int_normc-16_percentile-10.0_epsilon-0.05.npz'
dst = 'attack/imdb_attack_clean-centroid_int_normc-16_percentile-10.0_epsilon-0.05.npz'
copyfile(src, dst)

src = 'mnist_slab_bounds/attack/mnist_17_attack_clean-centroid_slab_normc-0.8_epsilon-0.3.npz'
dst = 'attack/mnist_17_attack_clean-centroid_slab_normc-0.8_epsilon-0.3.npz'
copyfile(src, dst)


