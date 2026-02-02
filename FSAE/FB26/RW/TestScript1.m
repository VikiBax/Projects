%% RW Surrogate Optimization (Ridge + k-fold CV + kNN Trust Penalty + PSO)

%% Clear
clear; clc;

%% Load data
raw_csv  = "01_71.csv";
raw_data = readmatrix(raw_csv);

% Inputs: AoA1 AoA2 AoA3 X1 X2 Z1 Z2 (7 vars)
X_raw = double(raw_data(:, 6:12));

% Outputs: CarL CarD RWLift RWDrag
Y      = double(raw_data(:, 14:17));
yLift  = Y(:,3);   % RW lift
yDrag  = Y(:,4);   % RW drag

%% Normalization stats (training)
mu = mean(X_raw, 1);
sg = std(X_raw, [], 1);
sg(sg == 0) = 1;

%% Quadratic features (normalized inside)
X_poly = polyfeatures(X_raw, mu, sg);

%% -------------------- 1) K-FOLD CV (manual, no crossval) --------------------
k = 5;
lambda = 1e-3;

cvp = cvpartition(size(X_poly,1), 'KFold', k);

yhat_cv = nan(size(yLift));
for fold = 1:k
    tr = training(cvp, fold);
    te = test(cvp, fold);

    mdl = fitrlinear(X_poly(tr,:), yLift(tr), ...
        'Learner','leastsquares', ...
        'Regularization','ridge', ...
        'Lambda', lambda);

    yhat_cv(te) = predict(mdl, X_poly(te,:));
end

rmse = sqrt(mean((yhat_cv - yLift).^2));
mae  = mean(abs(yhat_cv - yLift));
lift_range = max(yLift) - min(yLift);

disp('--- CV Diagnostics (Lift) ---');
disp(['k-fold k = ', num2str(k)]);
disp(['Lambda   = ', num2str(lambda)]);
disp(['RMSE     = ', num2str(rmse)]);
disp(['MAE      = ', num2str(mae)]);
disp(['Range    = ', num2str(min(yLift)), ' to ', num2str(max(yLift)), ...
      ' (range=', num2str(lift_range), ')']);
disp(['RMSE / range = ', num2str(rmse / max(lift_range, eps))]);

%% -------------------- 2) Train FINAL models on ALL data --------------------
model_lift = fitrlinear(X_poly, yLift, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda', lambda);

model_drag = fitrlinear(X_poly, yDrag, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda', lambda);

% Training residual scale (for optional "sanity cap")
yhat_train = predict(model_lift, X_poly);
resid = yLift - yhat_train;
sig_resid = std(resid);
ymax_reasonable = max(yLift) + 2*sig_resid;   % optional cap

%% -------------------- 3) kNN TRUST SETTINGS --------------------
% Work in normalized space for distance computation
Xn_train = (X_raw - mu) ./ sg;
Xn_train(~isfinite(Xn_train)) = 0;

% Nearest-neighbor distances among training points
D = pdist2(Xn_train, Xn_train);
D(D==0) = inf;
nn_dist = min(D, [], 2);

% Trust threshold (TUNE THIS)
% More strict: 70-80; More permissive: 90-95
dmax = prctile(nn_dist, 80);

% Penalty weight (TUNE THIS)
gamma_pen = 500;

%% -------------------- 4) PSO OPTIMIZATION --------------------
lb = min(X_raw, [], 1);
ub = max(X_raw, [], 1);

options = optimoptions('particleswarm', ...
    'Display','iter', ...
    'MaxIterations', 500);

obj = @(x) obj_lift_trust(x, model_lift, mu, sg, Xn_train, dmax, gamma_pen, ymax_reasonable);

nvars = size(X_raw, 2);
[x_opt, fval] = particleswarm(obj, nvars, lb, ub, options);

%% -------------------- 5) Report optimum --------------------
phi_opt   = polyfeatures(x_opt, mu, sg);
lift_pred = predict(model_lift, phi_opt);
drag_pred = predict(model_drag, phi_opt);

xn_opt = (x_opt - mu) ./ sg;
xn_opt(~isfinite(xn_opt)) = 0;
d_opt = min(vecnorm(Xn_train - xn_opt, 2, 2));

disp(' ');
disp('--- PSO Result ---');
disp('Optimal inputs (raw):');
disp(x_opt);

disp(['Predicted Lift: ', num2str(lift_pred)]);
disp(['Predicted Drag: ', num2str(drag_pred)]);
disp(['Max observed Lift: ', num2str(max(yLift))]);
disp(['Pred/Max observed: ', num2str(lift_pred / max(yLift))]);

disp(['d_opt (NN distance): ', num2str(d_opt)]);
disp(['dmax (trust thresh): ', num2str(dmax)]);
disp(['gamma_pen: ', num2str(gamma_pen)]);
disp(['Objective fval (minimized): ', num2str(fval)]);

%% ===================== Local Functions =====================

function f = obj_lift_trust(x, model, means, stds, Xn_train, dmax, gamma_pen, ymax_reasonable)
    % PSO minimizes f. We want to maximize lift but penalize untrusted points.
    % f(x) = -min(Lhat, ymax_reasonable) + gamma * max(0, dNN-dmax)^2

    % Prediction
    phi  = polyfeatures(x, means, stds);
    Lhat = predict(model, phi);

    % Optional sanity cap against extreme hallucinations
    Lhat = min(Lhat, ymax_reasonable);

    % Trust distance in normalized space
    xn = (x - means) ./ stds;
    xn(~isfinite(xn)) = 0;
    d = min(vecnorm(Xn_train - xn, 2, 2));

    % Penalty
    penalty = gamma_pen * max(0, d - dmax)^2;

    % Objective
    f = -Lhat + penalty;
end


% ======== ACTIVE SAMPLING: pick new CFD points ========

N_pool = 20000;     % candidate pool size
N_new  = 4;        % how many new CFD designs to run next (batch size)
P_exploit = 1; 
P_explore = 0; 

% Split batch
N_exploit = round(P_exploit*N_new);
N_explore = round(P_explore*N_new);
N_spike   = N_new - N_exploit - N_explore;

% Candidate pool (raw space)
Xcand = lb + (ub - lb) .* rand(N_pool, numel(lb));

% Predict lift for candidates
Phi_cand = polyfeatures(Xcand, mu, sg);
Lhat = predict(model_lift, Phi_cand);

% Normalized candidates for distance
Xcand_n = (Xcand - mu) ./ sg;
Xcand_n(~isfinite(Xcand_n)) = 0;

% Nearest-neighbor distance to existing data
dNN = min(pdist2(Xcand_n, Xn_train), [], 2);

% --- Bucket A: Exploit (high lift, but TRUSTED) ---
% Only consider candidates reasonably near existing data
d_trust = prctile(nn_dist, 65);  % tighter than before
trusted_idx = find(dNN <= d_trust);

[~, ordA] = sort(Lhat(trusted_idx), 'descend');
pickA = trusted_idx(ordA(1:min(N_exploit, numel(ordA))));

% --- Bucket B: Explore (most sparse regions) ---
[~, ordB] = sort(dNN, 'descend');
pickB = ordB(1:N_explore);

% --- Bucket C: Spike killers (local clouds around top predictions) ---
% Take a few of the highest predicted points (even if untrusted), then perturb locally
[~, ordTop] = sort(Lhat, 'descend');
seeds = Xcand(ordTop(1:min(5, numel(ordTop))), :);

Xspike = [];
for s = 1:size(seeds,1)
    % 1–3% of variable range perturbation (tune)
    frac = 0.02;
    jitter = frac .* (ub - lb) .* randn(ceil(N_spike/size(seeds,1)), numel(lb));
    Xlocal = seeds(s,:) + jitter;
    Xlocal = max(min(Xlocal, ub), lb);
    Xspike = [Xspike; Xlocal];
end
Xspike = Xspike(1:N_spike, :);

% Combine selected points
X_next = [Xcand(pickA,:); Xcand(pickB,:); Xspike];

disp('Next CFD batch designs (raw inputs):');
disp(X_next);
