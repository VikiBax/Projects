%Clearing 
clear; clc;

% Load data from CSV file for training
data = readmatrix('TestData.csv');

% Extract inputs (columns 4-9) and outputs (column 12) for training
inputs = data(2:end, 4:9); % Second AoA, Gap Size, etc.
outputs = data(2:end, 12); % L/D values

% Normalize inputs
inputs_normalized = (inputs - mean(inputs)) ./ std(inputs);

% Generate polynomial features
poly_degree = 2;
inputs_poly = [];
for i = 1:size(inputs_normalized, 2)
    for j = i:size(inputs_normalized, 2)
        inputs_poly = [inputs_poly, inputs_normalized(:, i) .* inputs_normalized(:, j)];
    end
end
inputs_poly = [inputs_normalized, inputs_poly]; % Include original inputs

% Fit quadratic regression model
model_1 = fitlm(inputs_poly, outputs);

% Evaluate model performance
rsquared = model_1.Rsquared.Ordinary;
disp(['R^2 of Quadratic Regression Model: ', num2str(rsquared)]);

%% Linear Correlation Matrix 
% Original variable names
var_name = {'AOA2', 'GS2', 'GA2', 'AOA3', 'GS3', 'GA3'};

corr_lin = corr(inputs_normalized);

figure;
heatmap(var_name, var_name, corr_lin, ...
    'Colormap', parula, 'ColorbarVisible', 'on');


%% Guadratic Correlation Matrix

% Generate all combinations for interaction terms
interaction_names = {};
for i = 1:numel(var_name)
    for j = i:numel(var_name)
        if i ~= j 
            interaction_names = [interaction_names, [var_name{i}, '*', var_name{j}]];
        end
    end
end

% Combine original names, squared terms, and interaction terms
variable_names = [var_name, strcat(var_name, '^2'), interaction_names];


% Generate quadratic and interaction terms
inputs_quad = [];
for i = 1:size(inputs_normalized, 2)
    for j = i:size(inputs_normalized, 2)
        inputs_quad = [inputs_quad, inputs_normalized(:, i) .* inputs_normalized(:, j)];
    end
end
inputs_quad = [inputs_normalized, inputs_quad]; % Include original inputs

% Compute correlation matrix for quadratic and interaction terms
correlation_matrix_quad = corr(inputs_quad);

% Create a heatmap of the quadratic correlation matrix
figure;
heatmap(variable_names, variable_names, correlation_matrix_quad, ...
    'Colormap', parula, 'ColorbarVisible', 'on');

% Add title
title('Correlation Matrix for Quadratic Features');


%% 
% Define ranges for optimization
ranges1 = [
    14, 22;    % Second AoA
    15, 35;    % Second Gap Size
    115, 135;  % Second Gap Angle
    30, 40;    % Third AoA
    5, 12;     % Third Gap Size
    57, 72     % Third Gap Angle
];
ranges2 = [
    0, 25;    % Second AoA
    6.5, 39;    % Second Gap Size (.05 to .3 times fwc2)
    10, 100;  % Second Gap Angle
    45, 75;    % Third AoA
    3.2, 19.2;     % Third Gap Size (.05 to .3 times fwc3)
    10, 100     % Third Gap Angle
];
ranges = ranges1;

lb = ranges(:, 1); % Lower bounds
ub = ranges(:, 2); % Upper bounds

% Find optimal test case using global optimization
disp('Finding optimal test case using global optimization...');
options = optimoptions('particleswarm', 'Display', 'iter', 'MaxIterations', 500);
optimal_inputs_1 = particleswarm(@(x) -predict(model_1, createPolyFeatures((x - mean(inputs)) ./ std(inputs), poly_degree)), ...
    size(inputs, 2), lb, ub, options);

disp('Optimal test case using particleswarm:');
disp(optimal_inputs_1);

% Predict L/D for the optimal inputs
optimal_inputs_normalized = (optimal_inputs_1 - mean(inputs)) ./ std(inputs);
optimal_inputs_poly = createPolyFeatures(optimal_inputs_normalized, poly_degree);
predicted_LD_optimal = predict(model_1, optimal_inputs_poly);
disp(['Predicted L/D ratio for optimal inputs: ', num2str(predicted_LD_optimal)]);

