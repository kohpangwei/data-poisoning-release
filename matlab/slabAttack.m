function [bestLower, bestUpper, lower_bounds, upper_bounds] = slabAttack(name, epsilon, eta, rho_squared, quantile, solver, MAX_ITER)
    rho = sqrt(rho_squared)
    fprintf(1, 'generating slab attack (poisoned means)\n');
    fprintf(1, 'parameters settings:\n');
    fprintf(1, '\tepsilon = %.3f | eta = %.4f | quantile = %.3f\n', epsilon, eta, quantile);

    rootPath = './datasets';
    load(sprintf('%s/%s/%s_data.mat', rootPath, name, name));
    [N_train, N_test, d, mus, probs, r_sphere, r_slab, r_ones] = processDataLight(X_train, y_train, X_test, y_test, quantile);
    destination = sprintf('%s_attack_eps%02d_quantile%02d_rho_slab_v7', name, round(100*epsilon), round(100*quantile));

    z = zeros(d,1);
    z_bias = 0;
    theta = zeros(d,1);
    bias = 0;
    freqUpperCheck = min(round(0.126 * MAX_ITER), 200)
    metadata = {};
    numUpper = 0;
    upper_bounds = [];
    lower_bounds = [];

    NUM_K = 10; % keep NUM_K best attacks found
    N_pert = round(epsilon * N_train);
    bestX = zeros(NUM_K, N_pert, d); % X_pert for best attacks
    bestY = zeros(NUM_K, N_pert); % y_pert for best attacks
    bestV = -inf * ones(NUM_K, 3); % claimed lower bound for best attacks (total, clean, poisoned)
    
    % initial step
    lambda = 1 / eta
    [g_c, L_c, dbias_c] = nabla_Loss(X_train, y_train, theta, bias);
    z = z - g_c;
    z_bias = z_bias - dbias_c;
    lambda = max(lambda, sqrt(norm(z,2)^2 + z_bias^2) / rho)
    theta = z / lambda; 
    bias = z_bias / lambda;
    Rcum = 0.5 * norm(g_c,2)^2 / lambda; % regret from initial gradient step

    for iter = 1:MAX_ITER
        fprintf(1, '====== STARTING ITERATION %d ======\n', iter);
        %EigCheck = eig([mus theta]' * [mus theta]);
        %EigCheck
        %SvdCheck = svds([mus theta]);
        %SvdCheck
        
        % first do an upper bound attack
        % flag of 0 means we heuristically set probs_eps = epsilon * probs
        disp('compute upper bound'); tic;
        [~,~,val,X_eps,probs_eps] = upperBoundTrue(X_train, y_train, theta, bias, probs, mus, epsilon, r_slab, r_sphere, 0, solver);
        maxVal = val;
        for s=1:10
            [~,~,val,X_eps_cur,probs_eps_cur] = upperBoundTrue(X_train, y_train, theta, bias, probs, mus, epsilon, r_slab, r_sphere, 1, solver);
            if val > maxVal
                maxVal = val;
                X_eps = X_eps_cur;
                probs_eps = probs_eps_cur;
            end
        end
        [~, L_c, ~, acc_c] = nabla_Loss(X_train, y_train, theta, bias);
        norm_sq = norm(theta,2)^2 + bias^2;
        fprintf(1, '\tupper bound: %.4f (all) | %.4f (clean) | %.4f (poisoned)\n', L_c + maxVal, L_c, maxVal / epsilon);
        fprintf(1, '\taccuracy: %.4f (clean)\n', acc_c);
        toc;
        xs = zeros(d,2);
        xs(:,1) = X_eps(:,1);
        xs(:,2) = X_eps(:,3);
        fprintf(1, '\tval: %.4f\n', val); % loss of attack
        
        % compute gradients and print loss
        [g_c, L_c, dbias_c, acc_c] = nabla_Loss(X_train, y_train, theta, bias);
        %SvdCheck2 = svd([mus g_c]);
        %SvdCheck2
        g_p = 0; L_p = 0; dbias_p = 0; acc_p = 0;
        for i=1:4
            [g_p_cur, L_p_cur, dbias_p_cur, acc_p_cur] = nabla_Loss(X_eps(:,i)', 2*(i<=2)-1, theta, bias);
            g_p = g_p + probs_eps(i) * g_p_cur;
            L_p = L_p + probs_eps(i) * L_p_cur;
            dbias_p = dbias_p + probs_eps(i) * dbias_p_cur;
            acc_p = acc_p + probs_eps(i) * acc_p_cur;
        end
        norm_sq = norm([theta;bias],2)^2;
        fprintf(1, 'loss: %.4f (clean) | %.4f (poisoned) | %.4f (norm_sq) | %.4f (all)\n', ...
            L_c, L_p / epsilon, ...
            norm_sq, L_c + L_p);
        fprintf(1, ' acc: %.4f (clean) | %.4f (poisoned)\n', acc_c, acc_p / epsilon);
        fprintf(1, 'norm of params: %.4f | bias: %.4f\n', norm(theta,2), bias);
        
        
        % do gradient update
        g = g_c + g_p;
        dbias = dbias_c + dbias_p;
        z = z - g;
        z_bias = z_bias - dbias;
        lambda = max(lambda, sqrt(norm(z,2)^2 + z_bias^2) / rho)
        theta = z / lambda; %(1/eta + iter * lambda);
        bias = z_bias / lambda; %(1/eta + iter * lambda);

        % output bound on the regret
        Rcum = Rcum + 0.5 * (norm(g,2)^2 + dbias^2) / lambda; %(1/eta + iter * lambda);
        fprintf(1, '\nAVERAGE REGRET after %d iterations: %.4f + %.4f |theta|_2^2\n\n', iter, Rcum / iter, 0.5 / (eta * iter));

        % every 10 iterations, try training against the current attack to
        % see how we do
        if (iter < 50 && mod(iter, 3) == 0) || (iter >= 50 && mod(iter, 10) == 0) || iter == MAX_ITER
            fprintf(1, 'Checking lower bound...\n');
            y_eps = [1 1 -1 -1];
            N_pert = round(epsilon * N_train);
            choices = mnrnd(1, probs_eps / sum(probs_eps), N_pert); %N_pert x 4
            X_pert_t = choices * X_eps';
            y_pert_t = choices * y_eps';
            N_tot = N_train + N_pert;
            [L_t, ~, theta_pert, bias_pert] = trainRDA2([X_train;X_pert_t], [y_train;y_pert_t], eta/(1+epsilon), N_tot, d, inf, 10, rho, 0);
            L_t = (1+epsilon) * L_t;
            [~, L_c, ~, acc_c] = nabla_Loss(X_train, y_train, theta_pert, bias_pert);
            [~, L_p, ~, acc_p] = nabla_Loss(X_pert_t, y_pert_t, theta_pert, bias_pert);
            L = L_c; % + epsilon * L_p; % + 0.5 * lambda * (norm(theta_pert,2)^2 + bias_pert)^2;
            fprintf(1, '\n\t************************************************************************\n');
            fprintf(1, '\t** LOWER BOUND: %.4f %.4f (full), %.4f (clean), %.4f (poisoned)\n', L_t, L, L_c, L_p);
            fprintf(1, '\t**  ACCURACIES: %.4f (clean), %.4f (poisoned)\n', acc_c, acc_p);
            fprintf(1, '\t**************************************************************************\n\n');
            lower_bounds = [lower_bounds; L];
            if L > bestV(NUM_K,1)
                bestV(NUM_K,:) = [L L_c L_p];
                bestX(NUM_K,:,:) = X_pert_t;
                bestY(NUM_K,:) = y_pert_t;
                k = NUM_K;
                while k > 1
                    if bestV(k,1) > bestV(k-1,1)
                        [bestV(k,:), bestV(k-1,:)] = deal(bestV(k-1,:), bestV(k,:));
                        [bestX(k,:,:), bestX(k-1,:,:)] = deal(bestX(k-1,:,:), bestX(k,:,:));
                        [bestY(k,:), bestY(k-1,:)] = deal(bestY(k-1,:), bestY(k,:));
                        k = k-1;
                    else
                        break;
                    end
                end
            end
        end
        if mod(iter, freqUpperCheck) == 0 || iter == MAX_ITER
            numUpper = numUpper + 1;
            S = 200;
            fprintf(1, 'Generating true upper bound (%d trials)...\n', S);
            maxVal = -inf;
            for s=1:S
                [~,~,val] = upperBoundTrue(X_train, y_train, theta, bias, probs, mus, epsilon, r_slab, r_sphere, 1, solver);
                maxVal = max(val, maxVal);
            end
            [~, L_c, ~, acc_c] = nabla_Loss(X_train, y_train, theta, bias);
            norm_sq = norm(theta,2)^2 + bias^2;
            fprintf(1, '\tTRUE UPPER BOUND: %.4f (all) | %.4f (clean) | %.4f (poisoned)\n', L_c + maxVal, L_c, maxVal / epsilon);
            fprintf(1, '\tACCURACY: %.4f (clean)\n', acc_c);

            upper_bounds = [upper_bounds; L_c + maxVal];
            metadata{numUpper,1} = struct('L_c', L_c, 'L_p', maxVal/epsilon, 'acc_c', acc_c, 'norm_theta', norm(theta,2), 'bias', bias);
        end

        if iter == freqUpperCheck || mod(iter, 50) == 0
          fprintf(1, 'SAVING INTERMEDIATE OUTPUT\n');
          bestLower = max(lower_bounds);
          bestUpper = min(upper_bounds);
          fprintf(1, 'best lower bound: %.4f\n', bestLower);
          fprintf(1, 'best upper bound: %.4f\n', bestUpper);
   
          Ravg = Rcum / iter;
          Ravg_norm = 0.5 / (eta * iter);
          metadata_final = metadata{numUpper,1};
          %destination = sprintf('%s/%s/%s_attack_eps%02d_quantile%02d_rho_slab_v7', rootPath, name, name, round(100*epsilon), round(100*quantile));
          save(destination, 'X_train', 'X_test', 'y_train', 'y_test', ...
              'bestLower', 'bestUpper', 'quantile', ...
              'bestX', 'bestY', 'bestV', 'N_pert', 'NUM_K', ...
              'theta', 'bias', 'Rcum', 'Ravg', 'Ravg_norm', 'MAX_ITER', ...
              'epsilon', 'eta', 'rho', 'lambda', 'metadata', 'metadata_final');
          fprintf(1, 'saved variables to %s\n', destination);
        end
    end
    bestLower = max(lower_bounds);
    bestUpper = min(upper_bounds);
    fprintf(1, 'best lower bound: %.4f\n', bestLower);
    fprintf(1, 'best upper bound: %.4f\n', bestUpper);
   
    Ravg = Rcum / MAX_ITER;
    Ravg_norm = 0.5 / (eta * MAX_ITER);
    metadata_final = metadata{numUpper,1};
    save(destination, 'X_train', 'X_test', 'y_train', 'y_test', ...
        'bestLower', 'bestUpper', 'quantile', ...
        'bestX', 'bestY', 'bestV', 'N_pert', 'NUM_K', ...
        'theta', 'bias', 'Rcum', 'Ravg', 'Ravg_norm', 'MAX_ITER', ...
        'epsilon', 'eta', 'rho', 'lambda', 'metadata', 'metadata_final');
    fprintf(1, 'saved variables to %s\n', destination);
    
end
