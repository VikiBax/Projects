%% 11_gp_blocked_cv.m
% Blocked CV for GP: train on early indices, test on late indices (and reverse)

clear; clc; close all;

raw_csv  = "01_89.csv";
raw_data = readmatrix(raw_csv);
raw_data = raw_data(1:87,:);

Xraw = raw_data(:, 6:13);
Y    = raw_data(:, 14:17);
yLift = Y(:,3);
yDrag = Y(:,4);

valid = all(isfinite(Xraw),2) & isfinite(yLift) & isfinite(yDrag);
Xraw = Xraw(valid,:);
yLift = yLift(valid);
yDrag = yDrag(valid);

N = size(Xraw,1);

% Normalize inputs
mu  = mean(Xraw,1);
sig = std(Xraw,0,1);
sig(sig==0)=1;
Xn = (Xraw - mu)./sig;

% Define blocks (edit breakpoint)
bp = 79;
tr1 = (1:bp)';     te1 = (bp+1:N)';
tr2 = (bp+1:N)';   te2 = (1:bp)';

kernel = "ardmatern52";
basis  = "constant";

% Try a few Sigma (noise floors) — this is usually key for CFD
sigmaList = [1, 3, 5, 10, 15];  % Newtons-ish; adjust based on your noise intuition

fprintf("N=%d | Block split: train 1:%d test %d:%d\n", N, bp, bp+1, N);

for s = 1:numel(sigmaList)
    sigY = sigmaList(s);

    % Train early -> test late
    mL1 = fitrgp(Xn(tr1,:), yLift(tr1), 'KernelFunction',kernel, 'BasisFunction',basis, ...
        'Sigma', sigY, 'ConstantSigma', true, 'FitMethod','exact', 'PredictMethod','exact');

    mD1 = fitrgp(Xn(tr1,:), yDrag(tr1), 'KernelFunction',kernel, 'BasisFunction',basis, ...
        'Sigma', sigY, 'ConstantSigma', true, 'FitMethod','exact', 'PredictMethod','exact');

    yhatL1 = predict(mL1, Xn(te1,:));
    yhatD1 = predict(mD1, Xn(te1,:));

    % Train late -> test early
    mL2 = fitrgp(Xn(tr2,:), yLift(tr2), 'KernelFunction',kernel, 'BasisFunction',basis, ...
        'Sigma', sigY, 'ConstantSigma', true, 'FitMethod','exact', 'PredictMethod','exact');

    mD2 = fitrgp(Xn(tr2,:), yDrag(tr2), 'KernelFunction',kernel, 'BasisFunction',basis, ...
        'Sigma', sigY, 'ConstantSigma', true, 'FitMethod','exact', 'PredictMethod','exact');

    yhatL2 = predict(mL2, Xn(te2,:));
    yhatD2 = predict(mD2, Xn(te2,:));

    % Metrics
    rmse = @(a,b) sqrt(mean((a-b).^2));
    r2   = @(a,b) 1 - sum((a-b).^2)/sum((a-mean(a)).^2);

    fprintf("\nSigma=%g\n", sigY);
    fprintf("  Lift  train early->late: RMSE=%.3f  R2=%.3f\n", rmse(yLift(te1), yhatL1), r2(yLift(te1), yhatL1));
    fprintf("  Lift  train late ->early: RMSE=%.3f  R2=%.3f\n", rmse(yLift(te2), yhatL2), r2(yLift(te2), yhatL2));
    fprintf("  Drag  train early->late: RMSE=%.3f  R2=%.3f\n", rmse(yDrag(te1), yhatD1), r2(yDrag(te1), yhatD1));
    fprintf("  Drag  train late ->early: RMSE=%.3f  R2=%.3f\n", rmse(yDrag(te2), yhatD2), r2(yDrag(te2), yhatD2));
end
