import IPython
import numpy as np

import os


def poison_with_influence_proj_gradient_step(model, test_idx, indices_to_poison, 
    projection_fn,
    step_size=0.01,    
    shrink_towards='cluster_center',
    loss_type='normal_loss',
    force_refresh=True, 
    test_description=None,
    output_root=None):
    """
    Returns poisoned_X_train, a subset of model.data_sets.train (marked by indices_to_poison)
    that has been modified by a single gradient step.
    """

    data_sets = model.data_sets

    if test_description is None:
        test_description = test_idx
    grad_filename = os.path.join(output_root, 'grad_influence_wrt_input_val_%s_testidx_%s.npy' % (model.model_name, test_description))

    if (force_refresh == False) and (os.path.exists(grad_filename)):
        grad_influence_wrt_input_val = np.load(grad_filename)
    else:
        grad_influence_wrt_input_val = model.get_grad_of_influence_wrt_input(
            indices_to_poison, 
            test_idx, 
            verbose=False,
            force_refresh=force_refresh,
            test_description=test_description,
            loss_type=loss_type)    

    poisoned_X_train = data_sets.train.x[indices_to_poison, :]
    poisoned_X_train -= step_size * grad_influence_wrt_input_val

    poisoned_labels = data_sets.train.labels[indices_to_poison]        
    poisoned_X_train = projection_fn(poisoned_X_train, poisoned_labels)

    return poisoned_X_train 


def iterative_attack(
    model, 
    indices_to_poison, 
    test_idx, 
    test_description=None,
    step_size=0.01, 
    num_iter=10,
    loss_type='normal_loss',
    projection_fn=None,
    output_root=None,
    stop_after=3): 

    largest_test_loss = 0
    stop_counter = 0

    print('Test idx: %s' % test_idx)

    np.save(os.path.join(output_root, '%s_indices' % model.model_name), indices_to_poison)
    np.savez(os.path.join(output_root, '%s_x_iter-0' % (model.model_name)), 
        poisoned_X_train=model.data_sets.train.x, 
        Y_train=model.data_sets.train.labels)

    for attack_iter in range(num_iter):
        print('*** Iter: %s' % attack_iter)

        # Create modified training dataset        
        old_X_train = np.copy(model.data_sets.train.x)
        poisoned_X_train_subset = poison_with_influence_proj_gradient_step(
            model, 
            test_idx, 
            indices_to_poison,
            projection_fn,
            step_size=step_size,
            loss_type=loss_type,
            force_refresh=True, 
            test_description=test_description,
            output_root=output_root)                
     
        poisoned_X_train = np.copy(model.data_sets.train.x)
        poisoned_X_train[indices_to_poison, :] = poisoned_X_train_subset

        # Measure some metrics on what the gradient step did
        labels = model.data_sets.train.labels
        dists_sum = 0.0
        poisoned_dists_sum = 0.0
        poisoned_mask = np.array([False] * len(labels), dtype=bool)
        poisoned_mask[indices_to_poison] = True
        for y in set(labels):
            cluster_center = np.mean(poisoned_X_train[labels == y, :], axis=0)
            dists = np.linalg.norm(poisoned_X_train[labels == y, :] - cluster_center, axis=1)
            dists_sum += np.sum(dists)

            poisoned_dists = np.linalg.norm(poisoned_X_train[(labels == y) & (poisoned_mask), :] - cluster_center, axis=1)
            poisoned_dists_sum += np.sum(poisoned_dists)

        dists_mean = dists_sum / len(labels)
        poisoned_dists_mean = poisoned_dists_sum / len(indices_to_poison)

        dists_moved = np.linalg.norm(old_X_train[indices_to_poison, :] - poisoned_X_train[indices_to_poison, :], axis=1)
        print('Average distance to cluster center (overall): %s' % dists_mean)
        print('Average distance to cluster center (poisoned): %s' % poisoned_dists_mean)
        print('Average diff in X_train among poisoned indices = %s' % np.mean(dists_moved))
        print('Fraction of 0 gradient points: %s' % np.mean(dists_moved == 0))
        print('Average distance moved by points that moved: %s' % np.mean(dists_moved[dists_moved > 0]))
        
        # Update training dataset
        model.update_train_x(poisoned_X_train)

        # Retrain model
        model.train()

        if (attack_iter + 1) % 40 == 0:

            # Calculate test loss
            test_loss = model.sess.run(model.loss_no_reg, feed_dict=model.all_test_feed_dict)
            if largest_test_loss < test_loss:
                largest_test_loss = test_loss

                np.savez(os.path.join(output_root, '%s_attack' % (model.model_name)), 
                    poisoned_X_train=poisoned_X_train, 
                    Y_train=model.data_sets.train.labels,
                    attack_iter=attack_iter + 1)

                stop_counter = 0
            else:
                stop_counter += 1

            if stop_counter >= stop_after:
                break


