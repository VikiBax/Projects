function phi = polyfeatures(x, mu, sigma)
% Quadratic feature mapping function
% x     : parameter vector
% mu    : mean
% sigma : standard deviation
% phi   : Polyfeatured output

x     = x(:)';
mu    = mu(:)';
sigma = sigma(:)';

n = numel(x);

% Normalize
z = (x - mu) ./ sigma;

% Feature count
nCross = n*(n-1)/2;
p = 1 + n + n + nCross;

phi = zeros(1, p);
idx = 1;

% Bias
phi(idx) = 1;
idx = idx + 1;

% Linear terms
for i = 1:n
    phi(idx) = z(i);
    idx = idx + 1;
end

% Squared terms
for i = 1:n
    phi(idx) = z(i)^2;
    idx = idx + 1;
end

% Cross terms
for i = 1:n-1
    for j = i+1:n
        phi(idx) = z(i) * z(j);
        idx = idx + 1;
    end
end

end
