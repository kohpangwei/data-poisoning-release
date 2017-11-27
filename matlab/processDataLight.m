function [N_train, N_test, d, mus, probs, r_sphere, r_slab, r_ones] = processDataLight(X_train, y_train, X_test, y_test, qq)
  N_train = size(X_train, 1);
  N_test = size(X_test, 1);
  d = size(X_train, 2);
  assert(d == size(X_test, 2));  
  assert(N_train == size(y_train,1));
  assert(N_test == size(y_test,1));
  assert(all(y_train == 1 | y_train == -1));
  assert(all(y_test == 1 | y_test == -1));
  N_plus = sum(y_train == 1);
  N_minus = sum(y_train == -1);
  mu_plus = X_train' * (y_train == 1) / N_plus;
  mu_minus = X_train' * (y_train == -1) / N_minus;
  mus = [mu_plus mu_minus];
  p_plus = N_plus / N_train;
  p_minus = N_minus / N_train;
  probs = [p_plus p_minus];

  tau_mean = computeTau(X_train, y_train, @(x,y) norm(x - mus(:, (3-y)/2), 2), 0);
  r_sphere = [quantile(tau_mean(:,1), qq) quantile(tau_mean(:,2), qq)];
  tau_slab = computeTau(X_train, y_train, @(x,y) abs(dot(x-mus(:, (3-y)/2), mus(:,1) - mus(:,2))), 0);
  r_slab = [quantile(tau_slab(:,1), qq) quantile(tau_slab(:,2), qq)];
  tau_ones = computeTau(X_train, y_train, @(x,y) abs(dot(x-mus(:, (3-y)/2), ones(d,1))), 0);
  r_ones = [quantile(tau_ones(:,1), qq) quantile(tau_ones(:,2), qq)];

end