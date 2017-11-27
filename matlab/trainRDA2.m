function [loss, acc, theta, bias] = trainRDA2(X, y, eta, N, d, Nmax, numIters, rho, verbose)
    z = zeros(d,1);
    z_bias = 0;
    theta = zeros(d,1);
    bias = 0;
    stepNum = 0;
    lambda = 1/eta;
    Rcum = 0;
    for iter=1:numIters
        totalLoss = 0;
        totalAcc = 0;
        pi = randperm(N);
        for ii=1:min(N,Nmax)
            stepNum = stepNum + 1;
            i = pi(ii);
            margin = y(i) * (X(i,:) * theta + bias);
            totalLoss = totalLoss + max(1-margin, 0); %0.5 * max(1-margin, 0)^2;
            totalAcc = totalAcc + (margin > 0);
            if margin < 1
                gradient = y(i) * X(i,:)'; % * (1-margin);
                gradient_bias = y(i);
                z = z + gradient;
                z_bias = z_bias + gradient_bias;
                lambda = max(lambda, sqrt(norm(z,2)^2 + z_bias^2) / rho);
                theta = z / lambda; %(1/eta + stepNum * lambda);
                bias = z_bias / lambda; %(1/eta + stepNum * lambda);
                Rcum = Rcum + 0.5 * (norm(gradient,2)^2 + gradient_bias^2) / lambda;
            end
            if Rcum / stepNum > 0.5 * rho^2 / (eta * stepNum)
              eta = 0.5 * eta;
              lambda = max(lambda, 1/eta);
            end
            if verbose && mod(ii,100) == 0
                fprintf(1, 'avg loss (iter %d): %.4f (%.4f)\n', ii, totalLoss / ii, totalAcc / ii);
            end
        end
        fprintf(1, 'trainRDA2 Regret after iteration %d: %.4f + %.4f * |w|^2\n', iter, Rcum / stepNum, 0.5 / (eta * stepNum));
        loss = totalLoss / min(N,Nmax);
        acc = totalAcc / min(N,Nmax);
    end
    %loss = loss + 0.5 * lambda * (norm(theta, 2)^2 + bias^2);
end
