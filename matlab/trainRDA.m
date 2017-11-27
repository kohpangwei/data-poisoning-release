function [loss, acc, theta, bias] = trainRDA(X, y, eta, N, d, Nmax, numIters, rho, verbose)
    z = zeros(d,1);
    z_bias = 0;
    theta = zeros(d,1);
    bias = 0;
    stepNum = 0;
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
                theta = z / (1/eta + stepNum * lambda);
                bias = z_bias / (1/eta + stepNum * lambda);
            end
            if verbose && mod(ii,100) == 0
                fprintf(1, 'avg loss (iter %d): %.4f (%.4f)\n', ii, totalLoss / ii + 0.5 * lambda * (norm(theta, 2)^2 + bias^2), totalAcc / ii);
            end
        end
        loss = totalLoss / min(N,Nmax);
        acc = totalAcc / min(N,Nmax);
    end
    loss = loss + 0.5 * lambda * (norm(theta, 2)^2 + bias^2);
end
