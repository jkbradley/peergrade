%% Refereed BTL model, with learning via SDP
%
% Note: w is re-scaled to be 'theta' from our write-up.
%
% Parameters:
%  wdist  Distribution of w: uniform/linear/sqrtlinear
%  d      # students
%  n      # comparisons (measurements)
%  sigma  # stddev of Gaussian noise added to measurements
%  a,b    Grading ability parameters: g = a w + b
function [w_loss, w_obj, w_est_loss, w_est_obj, w_est0_loss, w_est0_obj, ...
          w_L2norm, w_est_L2norm, w_est_diff_L2normed, w_est0_L2norm, w_est0_diff_L2normed, ...
          eigengap_normed, runtime] ...
    = BTLSDP(wdist, d, n, sigma, a, b)

%%
% Create BTL model and data

S = d; % L1norm(w)
allow_self_grades = false;

% w(i) = score of student i
if strcmp(wdist, 'uniform')
    w = ones(d,1);
elseif strcmp(wdist, 'linear')
    w = (1:d)';
elseif strcmp(wdist, 'sqrtlinear')
    w = (sqrt(1:d))';
else
    error(['bad wdist parameter: ' wdist]);
end
w = w .* (S / norm(w,1));
% NOTE: W IS BEING RESCALED TO BE THETA FROM OUR WRITE-UP.
w = w ./ sqrt(d);
a = a * d; b = b * sqrt(d);

g = a * w + b; % grading abilities

% Data
comparisons = zeros(n,4);
for k = 1:n
    if allow_self_grades
        i = randi(d,1,1);
        tmp = randperm(d,3);
        if i == tmp(1)
            j = tmp(2); l = tmp(3);
        else
            j = tmp(1);
            if i == tmp(2)
                l = tmp(3);
            else
                l = tmp(2);
            end
        end
    else % no self-grades
        tmp = randperm(d,3);
        i = tmp(1); j = tmp(2); l = tmp(3);
    end
    y = g(i) * (w(j) - w(l)) + normrnd(0,sigma);
    comparisons(k,:) = [i j l y];
end

%% Run optimization
% Create E = [E_1, E_2, ..., E_k]
Es = cell(n,1);
for k = 1:n
    i = comparisons(k,1)+1; j = comparisons(k,2)+1; l = comparisons(k,3)+1; y = comparisons(k,4);
    E_k_i = [1   j   1    l    i   j   i    l   ];
    E_k_j = [j   1   l    1    j   i   l    i   ];
    E_k_s = [b/2 b/2 -b/2 -b/2 a/2 a/2 -a/2 -a/2];
    Es{k} = sparse(E_k_i, E_k_j, E_k_s, d+1,d+1, length(E_k_i));
end

% Create [-d; ones(d,1)] for sum constraint.
d_ones_constraint = [-S/sqrt(d); ones(d,1)];
d_zeros = zeros(d,1);

lambda = 8 * sigma * sqrt(d * log(d) / n);

tic;
cvx_begin
    variable W(d+1,d+1);
    expression val(n+1);
    val(1) = 0;
    for k = 1:n,
        val(k+1) = val(k) + square(comparisons(k,4) - sum(dot(W,Es{k})));
    end
    val(n+1) = val(n+1) / (2*n);
    minimize( val(n+1) + lambda * trace(W) );
    subject to
        W == semidefinite(d+1);
        W(1,:) >= 0;
        W(:,1) >= 0;
        sum(W(1,2:end)) == S/sqrt(d);
        sum(W(2:end,1)) == S/sqrt(d);
        W(1,1) == 1;
        W(2:end,:) * d_ones_constraint == d_zeros;
        d_ones_constraint' * W(:, 2:end) == d_zeros';
cvx_end
runtime = toc;

% Compute estimate of w.
w_est0 = W(2:end,1);
[w_est_two, w_est_eigval_two] = eigs(W,2);
w_est = w_est_two(2:end,1);
w_est = w_est .* sign(sum(sign(w_est)));
w_est0 = w_est0 .* sign(sum(sign(w_est0)));

% Rescale estimates of w to have the same L1norm.
w_est = w_est .* (norm(w,1) / norm(w_est,1));
w_est0 = w_est0 .* (norm(w,1) / norm(w_est0,1));

%% Results

w
w_est
w_est0
[w_loss, w_obj] = computeObjective(Es, comparisons, w, lambda);
[w_est_loss, w_est_obj] = computeObjective(Es, comparisons, w_est, lambda);
[w_est0_loss, w_est0_obj] = computeObjective(Es, comparisons, w_est0, lambda);
w_L2norm = norm(w,2);
w_est_L2norm = norm(w_est,2);
w_est0_L2norm = norm(w_est0,2);
w_est_diff_L2normed = norm(w - w_est, 2) / w_L2norm;
w_est0_diff_L2normed = norm(w - w_est0, 2) / w_L2norm;
eigengap_normed = (w_est_eigval_two(1) - w_est_eigval_two(2))/w_est_eigval_two(1);

end


function [loss, obj] = computeObjective(Es, comparisons, w, lambda)
    n = size(comparisons,1);
    W = [1; w] * [1; w]';
    loss = 0;
    for k = 1:n,
        loss = loss + square(comparisons(k,4) - sum(dot(W,Es{k})));
    end
    loss = loss / (2*n);
    obj = loss + lambda * trace(W);
end


