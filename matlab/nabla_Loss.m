function [gradient, loss, gradient_bias, acc] = nabla_Loss(X, y, theta, bias)
    N = size(X, 1);
    d = size(X, 2);
    yX = spdiags(y, 0, N, N) * X;
    margins = yX * theta + y * bias;
    loss = sum(max(1-margins, 0)) / N;
    gradient = -sum(yX(margins < 1, :), 1)' / N;
    gradient_bias = -sum(y(margins<1)) / N;
    acc = sum(margins > 0) / N + 0.5 * sum(margins == 0) / N;
end
