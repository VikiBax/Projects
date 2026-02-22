%% 09_rbf_tune_hyperparams_lift.m
% Grid search epsilon and ridge for RBF (Gaussian) using k-fold CV

clear; clc; close all;

raw_csv  = "01_89.csv";
raw_data = readmatrix(raw_csv);
raw_data = raw_data(1:89,:);

Xraw = raw_data(:, 6:13);
Y    = raw_data(:, 14:17);
y    = Y(:,3); % Lift

valid = all(isfinite(Xraw),2) & isfinite(y);
Xraw = Xraw(valid,:);
y    = y(valid);

N = size(Xraw,1);

% Normalize
mu  = mean(Xraw,1);
sig = std(Xraw,0,1);
sig(sig==0)=1;
Xn  = (Xraw - mu)./sig;

% Base distance scale
dmed = median(pdist(Xn));

% Search grids
ridgeList   = [1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1];
epsScaleList = [0.1 0.2 0.5 1 2 5 10]; % epsilon = (scale)/dmed

k = 8;
cvp = cvpartition(N, "KFold", k);

best = struct('rmse', Inf, 'r2', -Inf, 'ridge', NaN, 'eps', NaN);

for r = 1:numel(ridgeList)
    ridge = ridgeList(r);

    for s = 1:numel(epsScaleList)
        epsval = epsScaleList(s) / max(eps, dmed);

        yhat_cv = nan(N,1);

        for fold = 1:k
            tr = training(cvp, fold);
            te = test(cvp, fold);

            m = rbf_train(Xn(tr,:), y(tr), "gaussian", epsval, ridge);
            yhat_cv(te) = rbf_predict(m, Xn(te,:));
        end

        rmse = sqrt(mean((yhat_cv - y).^2));
        r2 = 1 - sum((yhat_cv - y).^2) / sum((y - mean(y)).^2);

        fprintf("ridge=%-8.1e eps=%-8.3g  RMSE=%8.3f  R2=%7.3f\n", ridge, epsval, rmse, r2);

        if rmse < best.rmse
            best.rmse  = rmse;
            best.r2    = r2;
            best.ridge = ridge;
            best.eps   = epsval;
        end
    end
end

disp("====================================");
disp("BEST (by RMSE):");
disp(best);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local functions (same as before)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function model = rbf_train(Xc, y, kernel, epsval, ridge)
    N = size(Xc,1);
    K = rbf_kernel_matrix(Xc, Xc, kernel, epsval);
    w = (K + ridge*eye(N)) \ y;

    model.Xc = Xc;
    model.w = w;
    model.kernel = kernel;
    model.epsval = epsval;
    model.ridge = ridge;
end

function yhat = rbf_predict(model, Xq)
    Kq = rbf_kernel_matrix(Xq, model.Xc, model.kernel, model.epsval);
    yhat = Kq * model.w;
end

function K = rbf_kernel_matrix(Xa, Xb, kernel, epsval)
    R = pdist2(Xa, Xb);
    switch lower(kernel)
        case "gaussian"
            K = exp(-(epsval * R).^2);
        otherwise
            error("Only gaussian used in this tuner.");
    end
end
