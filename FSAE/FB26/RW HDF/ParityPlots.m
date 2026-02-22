%% 01_parity_plots_quad_ridge.m
% Parity plots (Predicted vs CFD) for RW Lift + RW Drag
% Uses same quadratic feature map + ridge regression style as your code.

clear; clc; close all;

%% Import Data
raw_csv  = "01_87.csv";
raw_data = readmatrix(raw_csv);
raw_data = raw_data(1:87,:);

%% Inputs / Outputs
% Inputs: AoA1 ; AoA2 ; AoA3 ; X1 ; X2 ; Z1 ; Z2 ; offset
Xraw = raw_data(:, 6:13);

% Outputs: Car Lift ; Car Drag ; RW Lift ; RW Drag
Y = raw_data(:, 15:18);
yLift = Y(:,3); % RW Lift
yDrag = Y(:,4); % RW Drag

% Filter NaNs (safety)
valid = all(isfinite(Xraw),2) & isfinite(yLift) & isfinite(yDrag);
Xraw  = Xraw(valid,:);
yLift = yLift(valid,:);
yDrag = yDrag(valid,:);

N = size(Xraw,1);
fprintf("Using N = %d samples\n", N);

%% Normalizing stats (match your approach)
mu = mean(Xraw, 1);
sig = std(Xraw, 0, 1);
sig(sig == 0) = 1;

%% Quadratic poly features
Phi = polyfeatures_quad(Xraw, mu, sig);

%% Regression Models (ridge)
lambda = 1e-3;

modelLift = fitrlinear(Phi, yLift, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda', lambda);

modelDrag = fitrlinear(Phi, yDrag, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda', lambda);

%% In-sample predictions
yhatLift = predict(modelLift, Phi);
yhatDrag = predict(modelDrag, Phi);

%% Cross-validated predictions (k-fold)
k = 8;  % change if you want (5/10 are common)
cvp = cvpartition(N, 'KFold', k);

yhatLift_cv = nan(N,1);
yhatDrag_cv = nan(N,1);

for fold = 1:k
    tr = training(cvp, fold);
    te = test(cvp, fold);

    % IMPORTANT: keep mu/sig from FULL dataset to match your workflow
    % (If you want strict CV, we can recompute mu/sig per fold.)
    Phi_tr = polyfeatures_quad(Xraw(tr,:), mu, sig);
    Phi_te = polyfeatures_quad(Xraw(te,:), mu, sig);

    mL = fitrlinear(Phi_tr, yLift(tr), ...
        'Learner','leastsquares', 'Regularization','ridge', 'Lambda', lambda);

    mD = fitrlinear(Phi_tr, yDrag(tr), ...
        'Learner','leastsquares', 'Regularization','ridge', 'Lambda', lambda);

    yhatLift_cv(te) = predict(mL, Phi_te);
    yhatDrag_cv(te) = predict(mD, Phi_te);
end

%% Metrics helper
metricLine = @(y,yhat) struct( ...
    'RMSE', sqrt(mean((yhat - y).^2)), ...
    'MAE',  mean(abs(yhat - y)), ...
    'R2',   1 - sum((yhat-y).^2)/sum((y-mean(y)).^2) ...
);

mLift_in  = metricLine(yLift, yhatLift);
mDrag_in  = metricLine(yDrag, yhatDrag);
mLift_cv  = metricLine(yLift, yhatLift_cv);
mDrag_cv  = metricLine(yDrag, yhatDrag_cv);

disp("=== In-sample metrics ===");
disp(table(["Lift";"Drag"], ...
    [mLift_in.RMSE; mDrag_in.RMSE], ...
    [mLift_in.MAE;  mDrag_in.MAE], ...
    [mLift_in.R2;   mDrag_in.R2], ...
    'VariableNames', {'Target','RMSE','MAE','R2'}));

disp("=== k-fold CV metrics ===");
disp(table(["Lift";"Drag"], ...
    [mLift_cv.RMSE; mDrag_cv.RMSE], ...
    [mLift_cv.MAE;  mDrag_cv.MAE], ...
    [mLift_cv.R2;   mDrag_cv.R2], ...
    'VariableNames', {'Target','RMSE','MAE','R2'}));

%% Plot settings
mkParity = @(y,yhat,ttl) parity_plot(y,yhat,ttl);

%% Parity plots: In-sample
mkParity(yLift, yhatLift, "RW Lift Parity (In-sample)");
mkParity(yDrag, yhatDrag, "RW Drag Parity (In-sample)");

%% Parity plots: Cross-validated
mkParity(yLift, yhatLift_cv, sprintf("RW Lift Parity (%d-fold CV)", k));
mkParity(yDrag, yhatDrag_cv, sprintf("RW Drag Parity (%d-fold CV)", k));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function parity_plot(y, yhat, ttl)
    figure;
    scatter(y, yhat, 36, "filled");
    grid on; axis equal;
    xlabel("CFD (true)");
    ylabel("Surrogate (pred)");
    title(ttl);

    % 45-degree line
    lo = min([y; yhat]);
    hi = max([y; yhat]);
    hold on;
    plot([lo hi], [lo hi], "k--", "LineWidth", 1.25);

    % Metrics annotation
    rmse = sqrt(mean((yhat - y).^2));
    mae  = mean(abs(yhat - y));
    r2   = 1 - sum((yhat-y).^2)/sum((y-mean(y)).^2);

    txt = sprintf("RMSE = %.3g\nMAE  = %.3g\nR^2  = %.3f", rmse, mae, r2);
    xPos = lo + 0.05*(hi-lo);
    yPos = hi - 0.10*(hi-lo);
    text(xPos, yPos, txt, "BackgroundColor","w", "EdgeColor",[0.7 0.7 0.7]);
end

function PHI = polyfeatures_quad(X, mu, sigma)
    if isvector(X)
        x = X(:)';
        [phi, ~] = polyfeatures_quad_row_with_grad(x, mu, sigma);
        PHI = phi;
        return;
    end

    N = size(X,1);
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
    idx = idx + 1;

    % Linear terms
    for i = 1:n
        phi(idx) = z(i);
        dphi_dz(i, idx) = 1;
        idx = idx + 1;
    end

    % Squared terms
    for i = 1:n
        phi(idx) = z(i)^2;
        dphi_dz(i, idx) = 2*z(i);
        idx = idx + 1;
    end

    % Cross terms
    for i = 1:n-1
        for j = i+1:n
            phi(idx) = z(i) * z(j);
            dphi_dz(i, idx) = z(j);
            dphi_dz(j, idx) = z(i);
            idx = idx + 1;
        end
    end

    % Chain rule dz/dx
    invsigma = 1 ./ sigma;            % 1 x n
    dphi_dx = dphi_dz .* invsigma(:); % (n x p)
end
