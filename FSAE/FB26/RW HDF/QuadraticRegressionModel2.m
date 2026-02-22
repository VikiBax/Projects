%% Quadratic Regression with K-Fold 
%
%% Clear workspace 
clear; clc; 

%% Load Data 
raw_csv = "01_87.csv"; 
raw_data = readmatrix(raw_csv);

% Inputs - AoA1, AoA2, AoA3, X1, X2, Z1, Z2 
x_raw = raw_data(:, 6:13); 

% Outputs - Car_DF, Car_D, RW_DF, RW_Drag 
y = raw_data(:, 15:18); 

% Pick target output 
lift = y(:,3);
drag = y(:,4);

y_target = lift; 

%% Normalization 
means = mean(x_raw, 1);
stds  = std(x_raw, [], 1);

% Quadratic Features - normalized inside 
x_poly = polyfeatures(x_raw, means, stds);

%% K Fold analysis 
k = 10;            % number of partitions (i.e. n-1 for training, 1 for test)
lambda = 1e-3;     % term to increase smoothness at void peaks

kfold(x_poly, y_target,k, lambda);


%% Training models on Final Data 
model_lift = fitrlinear(x_poly, lift, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda', lambda);

model_drag = fitrlinear(x_poly, drag, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda', lambda);

model_target = fitrlinear(x_poly, y_target, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda', lambda);

%model_preformance(model_lift, x_poly, lift) 
%model_preformance(model_drag, x_poly, drag) 
%model_preformance(model_target, x_poly, y_target)

ymax_reasonable = ymax_cap(model_target,x_poly,y_target);

%% kNN 



%% Optimization Settings 

% ranges 
ub   = [11.12, 32.84, 62.98, 4.04, 8.92, 23.31, 15.46, 10];
lb   = [5.28, 22.91, 53.13, -4.86, -1 , 13.47, 6.51, 5]; 

% options 
options = optimoptions('particleswarm', ...
                       'Display','iter', ...
                       'MaxIterations', 10);
% new point sampling 
numsamples = 20; 

n_uncapped = round(.6 * numsamples); 
n_capped = numsamples - n_uncapped; 

opt_config = []; 

%% Uncapped PSO Optimization (rank + unique like MultiStart script)

% --- ranking/uniqueness controls (same idea as your first script) ---
nStarts   = 300;      % number of independent PSO runs (increase for more maxima discovery)
maxUnique = n_uncapped; 
uniqueTol = 1e-6;     % distance tolerance for uniqueness in RAW input space

% Storage for all PSO outcomes
Xall = zeros(nStarts, size(x_raw,2));
Yall = zeros(nStarts, 1);

% Objective for PSO (maximize model prediction => minimize negative)
objPSO = @(x) -predict(model_target, polyfeatures(x, means, stds));

for s = 1:nStarts
    xopt = particleswarm(objPSO, size(x_raw,2), lb, ub, options);

    Xall(s,:) = xopt(:).';
    Yall(s)   = predict(model_target, polyfeatures(xopt, means, stds));
end

% ---- sort by best predicted value (descending) ----
[Ysorted, idx] = sort(Yall, 'descend');
Xsorted = Xall(idx,:);

% ---- keep only unique maxima (same clustering logic as your first script) ----
Xuniq = [];
Yuniq = [];

for k = 1:size(Xsorted,1)
    if isempty(Xuniq)
        Xuniq = Xsorted(k,:);
        Yuniq = Ysorted(k);
    else
        d = vecnorm(Xuniq - Xsorted(k,:), 2, 2);
        if all(d > uniqueTol)
            Xuniq = [Xuniq; Xsorted(k,:)];
            Yuniq = [Yuniq; Ysorted(k)];
        end
    end

    if size(Xuniq,1) >= maxUnique
        break;
    end
end

% ---- display ranked unique results ----
disp('==============================================')
disp('Uncapped PSO: Top UNIQUE maxima (ranked by predicted y_target):')
T_uncapped = table((1:size(Xuniq,1))', Yuniq, 'VariableNames', {'Rank','PredictedValue'});
disp(T_uncapped)
disp('Inputs for each ranked maximum (rows correspond to Rank):')
disp(Xuniq)

% If you want to keep your old variable name:
opt_config = Xuniq;


