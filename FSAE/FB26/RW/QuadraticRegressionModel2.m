%% Quadratic Regression with K-Fold 
%
%% Clear workspace 
clear; clc; 

%% Load Data 
raw_csv = "01_57.csv"; 
raw_data = readmatrix(raw_csv);

% Inputs - AoA1, AoA2, AoA3, X1, X2, Z1, Z2 
x_raw = raw_data(:, 5:11); 

% Outputs - Car_DF, Car_D, RW_DF, RW_Drag 
y = raw_data(:, 13:16); 

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
ub = max(x_raw, [], 1); 
lb = min(x_raw, [], 1); 

% options 
options = optimoptions('particleswarm', ...
                       'Display','iter', ...
                       'MaxIterations', 10);
% new point sampling 
numsamples = 20; 

n_uncapped = round(.6 * numsamples); 
n_capped = numsamples - n_uncapped; 

opt_config = []; 

%% Uncapped PSO Optimization 

while(true)
    optimal_inputs = particleswarm(@(x) -predict(model_target, polyfeatures(x, means, stds)), ...
        size(x_raw, 2), lb, ub, options);
    opt_config = similarity(optimal_inputs, opt_config);
    if size(opt_config,1) == n_uncapped 
        break  
    end 
end 

%% Capped PSO Optimization 

