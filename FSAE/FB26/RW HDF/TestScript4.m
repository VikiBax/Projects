%% 04_try_huber_ridge_quad.m
% Quadratic features + (A) ridge baseline vs (B) robust Huber + ridge (IRLS)
% Includes option to exclude specific outlier indices.

clear; clc; close all;

%% -------------------- Load data --------------------
raw_csv  = "01_89.csv";
raw_data = readmatrix(raw_csv);
raw_data = raw_data(1:89,:);

Xraw = raw_data(:, 6:13);      % 8 vars
Y    = raw_data(:, 14:17);     % CarLift CarDrag RWLift RWDrag

yLift = Y(:,3);
yDrag = Y(:,4);

valid = all(isfinite(Xraw),2) & isfinite(yLift) & isfinite(yDrag);
Xraw  = Xraw(valid,:);
yLift = yLift(valid);
yDrag = yDrag(valid);

N = size(Xraw,1);
fprintf("Using N=%d samples\n", N);

%% Your outliers (from your run)
outlier_idx = [23 24 25 27 32 37 67 76];
outlier_idx = outlier_idx(outlier_idx>=1 & outlier_idx<=N);

mask_all = true(N,1);
mask_no  = true(N,1); mask_no(outlier_idx) = false;

%% settings
k = 8;
lambda_ridge = 1e-3;     % baseline ridge
lambda_rob   = 1e-2;     % robust ridge (try 1e-3 to 1e-1)
delta        = 1.5;      % Huber transition (in robust "sigma" units)
maxIter      = 50;
tol          = 1e-6;

run_case("ALL POINTS", mask_all);
run_case("EXCLUDE OUTLIERS", mask_no);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function run_case(tag, keepMask)
    fprintf("\n==================== %s ====================\n", tag);
    figureOffset = sum(double(tag)); % crude unique offset per tag

    raw_csv  = "01_89.csv";
    raw_data = readmatrix(raw_csv);
    raw_data = raw_data(1:89,:);

    Xraw = raw_data(:, 6:13);
    Y    = raw_data(:, 14:17);
    yL = Y(:,3);
    yD = Y(:,4);

    valid = all(isfinite(Xraw),2) & isfinite(yL) & isfinite(yD);
    Xraw = Xraw(valid,:);
    yL = yL(valid);
    yD = yD(valid);

    X = Xraw(keepMask,:);
    yL = yL(keepMask);
    yD = yD(keepMask);

    N = size(X,1);
    fprintf("N used = %d\n", N);

    % normalize
    mu = mean(X,1);
    sig = std(X,0,1); sig(sig==0)=1;

    Phi = polyfeatures_quad(X, mu, sig);   % includes bias column as Phi(:,1)=1

    % --- Baseline ridge (fitrlinear) ---
    mL = fitrlinear(Phi, yL, 'Learner','leastsquares', 'Regularization','ridge', 'Lambda', 1e-3);
    mD = fitrlinear(Phi, yD, 'Learner','leastsquares', 'Regularization','ridge', 'Lambda', 1e-3);

    yL_hat = predict(mL, Phi);
    yD_hat = predict(mD, Phi);

    fprintf("Baseline RIDGE (in-sample):\n");
    disp(metrics_table(yL,yL_hat,yD,yD_hat));

    % --- k-fold CV: baseline ridge ---
    k = 8;
    cvp = cvpartition(N,'KFold',k);
    yL_cv = nan(N,1); yD_cv = nan(N,1);

    for f=1:k
        tr = training(cvp,f); te = test(cvp,f);
        Phi_tr = polyfeatures_quad(X(tr,:), mu, sig);
        Phi_te = polyfeatures_quad(X(te,:), mu, sig);

        mL = fitrlinear(Phi_tr, yL(tr), 'Learner','leastsquares', 'Regularization','ridge', 'Lambda', 1e-3);
        mD = fitrlinear(Phi_tr, yD(tr), 'Learner','leastsquares', 'Regularization','ridge', 'Lambda', 1e-3);

        yL_cv(te) = predict(mL, Phi_te);
        yD_cv(te) = predict(mD, Phi_te);
    end

    fprintf("Baseline RIDGE (%d-fold CV):\n", k);
    disp(metrics_table(yL,yL_cv,yD,yD_cv));

    % --- Robust Huber + Ridge (IRLS) ---
    % We solve:  min sum_i w_i * (y - Xb)^2  + lambda*||b||^2
    % with Huber weights updated from residuals.
    fprintf("ROBUST HUBER+RIDGE (in-sample):\n");
    bL = huber_ridge(Phi, yL, 1e-2, 1.5, 50, 1e-6);
    bD = huber_ridge(Phi, yD, 1e-2, 1.5, 50, 1e-6);

    yL_hatR = Phi*bL;
    yD_hatR = Phi*bD;
    disp(metrics_table(yL,yL_hatR,yD,yD_hatR));

    % --- k-fold CV: robust huber+ridge ---
    yL_cvR = nan(N,1); yD_cvR = nan(N,1);
    for f=1:k
        tr = training(cvp,f); te = test(cvp,f);
        Phi_tr = polyfeatures_quad(X(tr,:), mu, sig);
        Phi_te = polyfeatures_quad(X(te,:), mu, sig);

        bL = huber_ridge(Phi_tr, yL(tr), 1e-2, 1.5, 50, 1e-6);
        bD = huber_ridge(Phi_tr, yD(tr), 1e-2, 1.5, 50, 1e-6);

        yL_cvR(te) = Phi_te*bL;
        yD_cvR(te) = Phi_te*bD;
    end

    fprintf("ROBUST HUBER+RIDGE (%d-fold CV):\n", k);
    disp(metrics_table(yL,yL_cvR,yD,yD_cvR));

        %% -------------------- Parity plots --------------------
    % Ridge
    parity_plot(yL, yL_hat,  sprintf("Ridge In-sample | Lift | %s", tag));
    parity_plot(yD, yD_hat,  sprintf("Ridge In-sample | Drag | %s", tag));

    parity_plot(yL, yL_cv,   sprintf("Ridge %d-fold CV | Lift | %s", k, tag));
    parity_plot(yD, yD_cv,   sprintf("Ridge %d-fold CV | Drag | %s", k, tag));

    % Robust Huber+Ridge
    parity_plot(yL, yL_hatR, sprintf("Huber+Ridge In-sample | Lift | %s", tag));
    parity_plot(yD, yD_hatR, sprintf("Huber+Ridge In-sample | Drag | %s", tag));

    parity_plot(yL, yL_cvR,  sprintf("Huber+Ridge %d-fold CV | Lift | %s", k, tag));
    parity_plot(yD, yD_cvR,  sprintf("Huber+Ridge %d-fold CV | Drag | %s", k, tag));

end

function T = metrics_table(yL,yLhat,yD,yDhat)
    mL = metrics(yL,yLhat);
    mD = metrics(yD,yDhat);
    T = table(["Lift";"Drag"], [mL.RMSE;mD.RMSE], [mL.MAE;mD.MAE], [mL.R2;mD.R2], ...
        'VariableNames',{'Target','RMSE','MAE','R2'});
end

function m = metrics(y,yhat)
    m.RMSE = sqrt(mean((yhat-y).^2));
    m.MAE  = mean(abs(yhat-y));
    m.R2   = 1 - sum((yhat-y).^2)/sum((y-mean(y)).^2);
end

function b = huber_ridge(X, y, lambda, delta, maxIter, tol)
    % X includes bias column already if you want it
    [N,p] = size(X);
    b = (X'*X + lambda*eye(p)) \ (X'*y); % init ridge

    r = y - X*b;
    s = 1.4826 * mad(r,1);  % robust scale
    if s < eps, s = 1; end

    for it=1:maxIter
        r = y - X*b;
        u = r / (s+eps);

        % Huber weights
        w = ones(N,1);
        big = abs(u) > delta;
        w(big) = delta ./ (abs(u(big))+eps);

        W = spdiags(w,0,N,N);

        b_new = (X'*W*X + lambda*eye(p)) \ (X'*W*y);

        if norm(b_new - b) / (norm(b)+eps) < tol
            b = b_new;
            return;
        end
        b = b_new;
    end
end

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
    x=x(:)'; mu=mu(:)'; sigma=sigma(:)'; sigma(sigma==0)=1;
    z=(x-mu)./sigma;
    n=numel(z);
    nCross=n*(n-1)/2;
    p=1+2*n+nCross;
    phi=zeros(1,p);
    idx=1;
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


function parity_plot(y, yhat, ttl)
    figure('Color','w'); 
    scatter(y, yhat, 36, 'filled'); grid on; axis equal;
    xlabel('CFD (true)'); ylabel('Surrogate (pred)'); title(ttl);

    lo = min([y; yhat]); 
    hi = max([y; yhat]);
    hold on;
    plot([lo hi], [lo hi], 'k--', 'LineWidth', 1.25);

    rmse = sqrt(mean((yhat - y).^2));
    mae  = mean(abs(yhat - y));
    r2   = 1 - sum((yhat-y).^2)/sum((y-mean(y)).^2);

    txt = sprintf("RMSE = %.3g\\nMAE  = %.3g\\nR^2  = %.3f", rmse, mae, r2);
    xPos = lo + 0.05*(hi-lo);
    yPos = hi - 0.10*(hi-lo);
    text(xPos, yPos, txt, 'BackgroundColor','w', 'EdgeColor',[0.7 0.7 0.7]);
end
