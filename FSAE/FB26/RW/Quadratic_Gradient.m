%% Quadratic Regression + Gradient-Based Optimization

%% Clearing Workspace
clear; clc;

%% Import Data
raw_csv  = "01_84.csv";
raw_data = readmatrix(raw_csv);
raw_data = raw_data(1:84,:);

%% Inputs / Outputs
% Inputs: AoA1 ; AoA2 ; AoA3 ; X1 ; X2 ; Z1 ; Z2 ; offset  (columns 5:11)
inputs_raw = raw_data(:, 6:13);

% Outputs: Car Lift ; Car Drag ; RW Lift ; RW Drag (columns 13:16)
outputs = raw_data(:, 15:18);

output_lift = outputs(:,3); % RW Lift
output_drag = outputs(:,4); % RW Drag

%% normalizing stats
input_means = mean(inputs_raw, 1);
input_stds  = std(inputs_raw, 0, 1);

% Safety: avoid division by zero if any column is constant
input_stds(input_stds == 0) = 1;

%% Quadratic poly features
inputs_poly = polyfeatures_quad(inputs_raw, input_means, input_stds);

%% Regression Models
model_lift = fitrlinear(inputs_poly, output_lift, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda', 1e-3);

model_drag = fitrlinear(inputs_poly, output_drag, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda', 1e-3);

%% Optimization Setup (Gradient-based)

% Bounds from observed data
ub   = [11.12, 32.84, 62.98, 4.04, 8.92, 23.31, 15.46, 10];
lb   = [5.28, 22.91, 53.13, -4.86, -1 , 13.47, 6.51, 5];
nvar = size(inputs_raw, 2);

% Choose which model to maximize
model = model_lift;

opts = optimoptions('fmincon', ...
    'Algorithm','interior-point', ...
    'SpecifyObjectiveGradient', true, ...
    'Display','iter', ...
    'MaxIterations', 500, ...
    'OptimalityTolerance', 1e-10, ...
    'StepTolerance', 1e-12);

% Objective: minimize negative prediction => maximize prediction
obj = @(x) objective_neg_predict_with_grad(x, model, input_means, input_stds);

% Build a baseline fmincon problem
x0 = mean([lb; ub], 1);  % center of box
problem = createOptimProblem('fmincon', ...
    'objective', obj, ...
    'x0', x0, ...
    'lb', lb, ...
    'ub', ub, ...
    'options', opts);

%% MultiStart to surface multiple local maxima
nStarts     = 1000;     % increase (100-500) for more local maxima discovery
maxUnique   = 15;      % how many unique local maxima to keep
uniqueTol   = 1e-6;    % distance tolerance for uniqueness in raw input space

ms = MultiStart('UseParallel', false, 'Display', 'off');
startPoints = RandomStartPointSet('NumStartPoints', nStarts);

[xbest, fbest, exitflag_best, output_best, solutions] = run(ms, problem, startPoints);

%% Collect and rank solutions (convert fbest -> predicted)
numS = numel(solutions);
X = zeros(numS, nvar);
Y = zeros(numS, 1);

for k = 1:numS
    X(k,:) = solutions(k).X(:).';
    % Predict using SAME polyfeatures
    phi = polyfeatures_quad(X(k,:), input_means, input_stds);
    Y(k) = predict(model, phi);
end

% Sort by highest predicted value
[YS, idx] = sort(Y, 'descend');
XS = X(idx,:);

% Keep only unique maxima (simple clustering by Euclidean distance)
Xuniq = [];
Yuniq = [];

for k = 1:size(XS,1)
    if isempty(Xuniq)
        Xuniq = XS(k,:);
        Yuniq = YS(k);
    else
        d = vecnorm(Xuniq - XS(k,:), 2, 2);
        if all(d > uniqueTol)
            Xuniq = [Xuniq; XS(k,:)]; %#ok<AGROW>
            Yuniq = [Yuniq; YS(k)];   %#ok<AGROW>
        end
    end
    if size(Xuniq,1) >= maxUnique
        break;
    end
end

%% Display results
disp('==============================================')
disp('Best local maximum found by MultiStart + fmincon:')
disp('Optimal inputs (xbest):')
disp(xbest)
disp(['Predicted value at xbest: ', num2str(-fbest)])
disp(['exitflag: ', num2str(exitflag_best)])
disp('==============================================')

disp('Top UNIQUE local maxima (ranked):')
T = table((1:size(Xuniq,1))', Yuniq, 'VariableNames', {'Rank','PredictedValue'});
disp(T)

disp('Inputs for each ranked local maximum (rows correspond to Rank):')
disp(Xuniq)

%% (Optional) verify best point prediction directly
phi_best = polyfeatures_quad(xbest, input_means, input_stds);
y_best   = predict(model, phi_best);
disp(['Sanity check predict(model, phi_best) = ', num2str(y_best)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f, g] = objective_neg_predict_with_grad(x, model, mu, sigma)
% Objective for fmincon:
%   f(x) = -prediction(x)
%   g(x) = gradient of f wrt x (analytic)

    x = x(:)'; % row

    % Feature vector and Jacobian dphi/dx
    [phi, dphi_dx] = polyfeatures_quad_row_with_grad(x, mu, sigma);

    % Linear model prediction y = Bias + phi * Beta
    beta = model.Beta;   % p x 1
    bias = model.Bias;   % scalar

    y = bias + phi * beta;      % scalar

    % dy/dx = (dphi/dx) * beta   where dphi/dx is (n x p)
    dydx = dphi_dx * beta;      % n x 1

    % Minimize negative of prediction
    f = -y;
    g = -dydx;                  % n x 1
end

function PHI = polyfeatures_quad(X, mu, sigma)
    if isvector(X)
        x = X(:)';
        [phi, ~] = polyfeatures_quad_row_with_grad(X(:)', mu, sigma);
        PHI = phi;
        return;
    end

    N = size(X,1);
    % Build first row to get p
    [phi1, ~] = polyfeatures_quad_row_with_grad(X(1,:), mu, sigma);
    p = numel(phi1);

    PHI = zeros(N, p);
    PHI(1,:) = phi1;

    for r = 2:N
        [PHI(r,:), ~] = polyfeatures_quad_row_with_grad(X(r,:), mu, sigma);
    end
end

function [phi, dphi_dx] = polyfeatures_quad_row_with_grad(x, mu, sigma)
    x     = x(:)';      % 1 x n
    mu    = mu(:)';     % 1 x n
    sigma = sigma(:)';  % 1 x n

    n = numel(x);

    % Normalize
    z = (x - mu) ./ sigma;

    % Feature count
    nCross = n*(n-1)/2;
    p = 1 + n + n + nCross;

    phi     = zeros(1, p);
    dphi_dz = zeros(n, p);

    idx = 1;

    % Bias term
    phi(idx) = 1;
    % dphi_dz(:,idx) already zeros
    idx = idx + 1;

    % Linear terms: z_i
    for i = 1:n
        phi(idx) = z(i);
        dphi_dz(i, idx) = 1;
        idx = idx + 1;
    end

    % Squared terms: z_i^2
    for i = 1:n
        phi(idx) = z(i)^2;
        dphi_dz(i, idx) = 2*z(i);
        idx = idx + 1;
    end

    % Cross terms: z_i * z_j for i<j
    for i = 1:n-1
        for j = i+1:n
            phi(idx) = z(i) * z(j);
            dphi_dz(i, idx) = z(j);
            dphi_dz(j, idx) = z(i);
            idx = idx + 1;
        end
    end

    % Chain rule: z_i = (x_i - mu_i)/sigma_i => dz_i/dx_i = 1/sigma_i
    invsigma = 1 ./ sigma;                  % 1 x n
    dphi_dx = dphi_dz .* invsigma(:);       % (n x p) elementwise scaling per row
end
