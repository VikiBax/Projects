%% Clearing Workspace
clear; clc; 

% Import Data 
raw_csv = "01_51.csv"; 
raw_data = readmatrix(raw_csv); 

%% Extracting Inputs and Outputs 

% AoA 1 ; AoA 2 ; AoA 3 ; X 1 ; X 2; Z 1 ; Z 2
inputs_raw = raw_data(:, 6:12); 

% Car Lift ; Car Drag ; RW Lift ; RW Drag
outputs = raw_data(:, 14:17); 

output_lift = outputs(:,3);
output_drag = outputs(:,4);

% Normalize inputs 
input_means = mean(inputs_raw);
input_stds = std(inputs_raw);
inputs_poly = polyfeatures(inputs_raw, input_means, input_stds);

%% Regression Model 
% Fit quadratic regression models 
model_lift = fitrlinear(inputs_poly, output_lift, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda', 1e-3);
model_drag = fitrlinear(inputs_poly, output_drag, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda', 1e-3);

%model_preformance(model_lift, inputs_poly, output_lift)
%model_preformance(model_drag, inputs_poly, output_drag)

%% Optimization 

% range for optimization 
ub = max(inputs_raw, [], 1);
lb = min(inputs_raw, [], 1);

% OPTIONS
model = model_lift;
options = optimoptions('particleswarm', ...
                       'Display','iter', ...
                       'MaxIterations', 500);

opt_config = []; 
i = 0;
max_loops = 1;

while(true)
    optimal_inputs = particleswarm(@(x) -predict(model, polyfeatures(x, input_means, input_stds)), ...
        size(inputs_raw, 2), lb, ub, options);
    opt_config = similarity(optimal_inputs, opt_config);
    i = i + 1;

    if size(opt_config,1) == 15 || i == max_loops
        break 
    end
end

disp('Optimal test case case using particleswarm:')
disp(optimal_inputs)

% predict lift for optimal 
optimal_inputs_poly = polyfeatures(optimal_inputs, input_means, input_stds);
prediction = predict(model, optimal_inputs_poly);
disp(['Predicted value for optimal inputs:', num2str(prediction) ])

