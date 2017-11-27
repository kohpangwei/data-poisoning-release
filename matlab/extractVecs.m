function V_full = extractVecs(G_full, G_partial, V_partial)
    % G_full is Graham matrix of inner products
    % G_partial is lower-right corner of G
    % V_partial is collection of vectors realizing G_partial
    n_full = size(G_full, 1);
    assert(n_full == size(G_full, 2));
    n_partial = size(G_partial, 1);
    n_missing = n_full - n_partial;
    assert(n_partial == size(G_partial, 2));
    assert(n_partial == size(V_partial, 2));
    d = size(V_partial, 1);
    
    [Proj_half, ~] = svd_lr(V_partial, 1e-6);
    G_11 = G_full(1:n_missing,1:n_missing);
    G_12 = G_full(1:n_missing,n_missing+1:n_full);
    G_22 = G_full(n_missing+1:n_full,n_missing+1:n_full);
    [U_22, D_22] = eig_lr(G_22, 1e-6);
    Gp_12 = G_12 * U_22;
    Gp_22_pinv_sqrt = sqrt(inv(D_22));
    Gp_schur_sqrt = Gp_12 * Gp_22_pinv_sqrt;
    Bp = Gp_schur_sqrt * Gp_22_pinv_sqrt';
    AAt = (G_11 - (Gp_schur_sqrt * Gp_schur_sqrt'));
    [U_a, D_a] = eig(AAt);
    A = U_a * sqrt(max(D_a,0));
    
    basis = randn(d,n_missing);
    basis = basis - Proj_half * (Proj_half' * basis);
    [basis, ~] = qr(basis, 0);
        
    V_missing = basis * A' + V_partial * U_22 * Bp';
        
    V_full = [V_missing V_partial];
    err = norm(G_full - (V_full'*V_full), 'inf');
    fprintf(1, 'err: %.4f\n', err);
        
end

function [U,D] = svd_lr(A, tol)
    [U,D,~] = svd(A, 'econ');
    active = diag(D) > tol*max(D(:));
    U = U(:,active);
    D = D(active,active);
end

function [U,D] = eig_lr(A, tol)
    [U,D] = eig((A+A')/2);
    active = diag(D)>tol*max(D(:));
    U = U(:, active);
    D = D(active,active);
end

