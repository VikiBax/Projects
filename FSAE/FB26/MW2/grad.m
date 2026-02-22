function dphi_dx = grad(x, mu, sigma)
% Gradient of quadratic feature map
% x     : parameter vector
% mu    : mean
% sigma : standard deviation
% dphi_dx(i,j) = d(phi_j)/d(x_i)

x     = x(:)';
mu    = mu(:)';
sigma = sigma(:)';

n = numel(x);

% Normalize
z = (x - mu) ./ sigma;

% Feature count
nCross = n*(n-1)/2;
p = 1 + n + n + nCross;

dphi_dz = zeros(n, p);
idx = 1;

% Bias (no gradient)
idx = idx + 1;

% Linear terms
for i = 1:n
    dphi_dz(i, idx) = 1;
    idx = idx + 1;
end

% Squared terms
for i = 1:n
    dphi_dz(i, idx) = 2*z(i);
    idx = idx + 1;
end

% Cross terms
for i = 1:n-1
    for j = i+1:n
        dphi_dz(i, idx) = z(j);
        dphi_dz(j, idx) = z(i);
        idx = idx + 1;
    end
end

% Chain rule: dz_i/dx_i = 1/sigma_i
dphi_dx = dphi_dz .* (1 ./ sigma(:));
end
