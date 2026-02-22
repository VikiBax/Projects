%% 03_try_new_surrogate_robust_quad.m
% Compare baseline quadratic ridge vs robust regression (Huber) on quad features
% Evaluate with/without your flagged outliers

clear; clc; close all;

%% -------------------- Load data --------------------
raw_csv  = "01_89.csv";
raw_data = readmatrix(raw_csv);
raw_data = raw_data(1:89,:);

Xraw = raw_data(:, 6:13);      % AoA1..AoA3, X1..X2, Z1..Z2, offset
Y    = raw_data(:, 14:17);     % CarLift, CarDrag, RWLift, RWDrag

yLift = Y(:,3);
yDrag = Y(:,4);

valid = all(isfinite(Xraw),2) & isfinite(yLift) & isfinite(yDrag);
Xraw = Xraw(valid,:);
yLift = yLift(valid);
yDrag = yDrag(valid);

N = size(Xraw,1);
fprintf("Using N=%d samples\n", N);

%% -------------------- Define your outliers --------------------
outlier_idx = [23 24 25 27 32 37 67 76]; % <-- from your printout
outlier_idx = outlier_idx(outlier_idx >= 1 & outlier_idx <= N);

mask_keep_all = true(N,1);
mask_keep_no  = true(N,1);
mask_keep_no(outlier_idx) = false;

%% -------------------- Settings --------------------
lambda = 1e-3;       % ridge for baseline
kfold  = 8;

% Optional: block split (train early -> test late)
splitIdx = 79;
useBlockSplit = (N > splitIdx);

labels = {'AoA1','AoA2','AoA3','X1','X2','Z1','Z2','offset'};

%% -------------------- Run both scenarios --------------------
run_case("ALL POINTS", mask_keep_all);
run_case("EXCLUDE OUTLIERS", mask_keep_no);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper: run one scenario
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function run_case(tag, keepMask)
    fprintf("\n==================== %s ====================\n", tag);

    raw_csv  = "01_89.csv";
    raw_data = readmatrix(raw_csv);
    raw_data = raw_data(1:89,:);

    Xraw = raw_data(:, 6:13);
    Y    = raw_data(:, 14:17);

    yLift = Y(:,3);
    yDrag = Y(:,4);

    valid = all(isfinite(Xraw),2) & isfinite(yLift) & isfinite(yDrag);
    Xraw = Xraw(valid,:);
    yLift = yLift(valid);
    yDrag = yDrag(valid);

    % apply keep mask (must match N after valid filter)
    X = Xraw(keepMask,:);
    yL = yLift(keepMask);
    yD = yDrag(keepMask);

    N = size(X,1);
    fprintf("N used = %d\n", N);

    % normalization
    mu = mean(X,1);
    sig = std(X,0,1); sig(sig==0) = 1;

    % quad features
    Phi = polyfeatures_quad(X, mu, sig);

    % -------------------- Model A: Quadratic Ridge (baseline) --------------------
    lambda = 1e-3;
    mL_ridge = fitrlinear(Phi, yL, 'Learner','leastsquares', 'Regularization','ridge', 'Lambda', lambda);
    mD_ridge = fitrlinear(Phi, yD, 'Learner','leastsquares', 'Regularization','ridge', 'Lambda', lambda);

    yL_hat = predict(mL_ridge, Phi);
    yD_hat = predict(mD_ridge, Phi);

    metL_in = metrics(yL, yL_hat);
    metD_in = metrics(yD, yD_hat);

    fprintf("Baseline RIDGE (in-sample):\n");
    disp(table(["Lift";"Drag"], [metL_in.RMSE; metD_in.RMSE], [metL_in.MAE; metD_in.MAE], [metL_in.R2; metD_in.R2], ...
        'VariableNames', {'Target','RMSE','MAE','R2'}));

    % k-fold CV for ridge
    kfold = 8;
    cvp = cvpartition(N, 'KFold', kfold);
    yL_cv = nan(N,1); yD_cv = nan(N,1);

    for f=1:kfold
        tr = training(cvp,f);
        te = test(cvp,f);

        Phi_tr = polyfeatures_quad(X(tr,:), mu, sig);
        Phi_te = polyfeatures_quad(X(te,:), mu, sig);

        mL = fitrlinear(Phi_tr, yL(tr), 'Learner','leastsquares', 'Regularization','ridge', 'Lambda', lambda);
        mD = fitrlinear(Phi_tr, yD(tr), 'Learner','leastsquares', 'Regularization','ridge', 'Lambda', lambda);

        yL_cv(te) = predict(mL, Phi_te);
        yD_cv(te) = predict(mD, Phi_te);
    end

    metL_cv = metrics(yL, yL_cv);
    metD_cv = metrics(yD, yD_cv);

    fprintf("Baseline RIDGE (%d-fold CV):\n", kfold);
    disp(table(["Lift";"Drag"], [metL_cv.RMSE; metD_cv.RMSE], [metL_cv.MAE; metD_cv.MAE], [metL_cv.R2; metD_cv.R2], ...
        'VariableNames', {'Target','RMSE','MAE','R2'}));

    % -------------------- Model B: Robust Regression (Huber) on quad features --------------------
    % This uses robustfit (Huber by default). It downweights outliers instead of removing them.
    % Add intercept automatically via robustfit.
    bL = robustfit(Phi(:,2:end), yL, 'huber'); % exclude explicit bias column (Phi(:,1)=1)
    bD = robustfit(Phi(:,2:end), yD, 'huber');

    yL_hatR = bL(1) + Phi(:,2:end)*bL(2:end);
    yD_hatR = bD(1) + Phi(:,2:end)*bD(2:end);

    metL_inR = metrics(yL, yL_hatR);
    metD_inR = metrics(yD, yD_hatR);

    fprintf("ROBUST (Huber) (in-sample):\n");
    disp(table(["Lift";"Drag"], [metL_inR.RMSE; metD_inR.RMSE], [metL_inR.MAE; metD_inR.MAE], [metL_inR.R2; metD_inR.R2], ...
        'VariableNames', {'Target','RMSE','MAE','R2'}));

    % k-fold CV for robust
    yL_cvR = nan(N,1); yD_cvR = nan(N,1);

    for f=1:kfold
        tr = training(cvp,f);
        te = test(cvp,f);

        Phi_tr = polyfeatures_quad(X(tr,:), mu, sig);
        Phi_te = polyfeatures_quad(X(te,:), mu, sig);

        bL = robustfit(Phi_tr(:,2:end), yL(tr), 'huber');
        bD = robustfit(Phi_tr(:,2:end), yD(tr), 'huber');

        yL_cvR(te) = bL(1) + Phi_te(:,2:end)*bL(2:end);
        yD_cvR(te) = bD(1) + Phi_te(:,2:end)*bD(2:end);
    end

    metL_cvR = metrics(yL, yL_cvR);
    metD_cvR = metrics(yD, yD_cvR);

    fprintf("ROBUST (Huber) (%d-fold CV):\n", kfold);
    disp(table(["Lift";"Drag"], [metL_cvR.RMSE; metD_cvR.RMSE], [metL_cvR.MAE; metD_cvR.MAE], [metL_cvR.R2; metD_cvR.R2], ...
        'VariableNames', {'Target','RMSE','MAE','R2'}));

    % -------------------- Quick Parity Plots (CV) --------------------
    parity_plot(yL, yL_cv,  sprintf("Ridge CV Parity (Lift) - %s", tag));
    parity_plot(yL, yL_cvR, sprintf("Robust CV Parity (Lift) - %s", tag));

    parity_plot(yD, yD_cv,  sprintf("Ridge CV Parity (Drag) - %s", tag));
    parity_plot(yD, yD_cvR, sprintf("Robust CV Parity (Drag) - %s", tag));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Metrics + plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function m = metrics(y, yhat)
    m.RMSE = sqrt(mean((yhat - y).^2));
    m.MAE  = mean(abs(yhat - y));
    m.R2   = 1 - sum((yhat-y).^2)/sum((y-mean(y)).^2);
end

function parity_plot(y, yhat, ttl)
    figure('Color','w');
    scatter(y, yhat, 36, 'filled'); grid on; axis equal;
    xlabel('CFD (true)'); ylabel('Surrogate (pred)'); title(ttl);
    lo = min([y; yhat]); hi = max([y; yhat]);
    hold on; plot([lo hi],[lo hi],'k--','LineWidth',1.25);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Quadratic feature map (same style as yours)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function PHI = polyfeatures_quad(X, mu, sigma)
    if isvector(X)
        x = X(:)';
        PHI = polyfeatures_quad_row(x, mu, sigma);
        return;
    end
    N = size(X,1);
    PHI = zeros(N, 1 + 2*size(X,2) + size(X,2)*(size(X,2)-1)/2);
    for i=1:N
        PHI(i,:) = polyfeatures_quad_row(X(i,:), mu, sigma);
    end
end

function phi = polyfeatures_quad_row(x, mu, sigma)
    x = x(:)'; mu = mu(:)'; sigma = sigma(:)'; sigma(sigma==0)=1;
    z = (x - mu)./sigma;
    n = numel(z);
    nCross = n*(n-1)/2;
    p = 1 + 2*n + nCross;
    phi = zeros(1,p);
    idx = 1;
    phi(idx)=1; idx=idx+1;
    phi(idx:idx+n-1)=z; idx=idx+n;
    phi(idx:idx+n-1)=z.^2; idx=idx+n;
    for i=1:n-1
        for j=i+1:n
            phi(idx)=z(i)*z(j);
            idx=idx+1;
        end
    end
end
