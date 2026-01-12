% Load data from CSV file
data = readmatrix('TestData2.csv');

% Extract inputs (columns 4-9) and outputs (column 12)
inputs = data(2:end, 4:9); % Second AoA, Gap Size, etc.
outputs = data(2:end, 12); % L/D values

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
ranges3 = [ 
    0, 25; 
    6.5, 39;
    10, 135;
    30, 75; 
    3.2, 19.2; 
    10, 100];
ranges = ranges2;

lb = ranges(:, 1); % Lower bounds
ub = ranges(:, 2); % Upper bounds

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

% Predict outputs for all test cases
predicted_outputs_1 = predict(model_1, inputs_poly);

% Evaluate model performance
rsquared = model_1.Rsquared.Ordinary;
disp(['R^2 of Quadratic Regression Model: ', num2str(rsquared)]);

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

% Visualization: Actual vs Predicted L/D with ±1% error bounds
figure;

% Scatter plot of actual vs predicted values
scatter(outputs, predicted_outputs_1, 'b', 'DisplayName', 'Quadratic Predictions');
hold on;

% Add x = y line (perfect prediction)
x = min(outputs):0.01:max(outputs);
plot(x, x, 'k--', 'DisplayName', 'x = y');

% Add ±1% error bounds
upper_bound = x * 1.01; % +1% line
lower_bound = x * 0.99; % -1% line
plot(x, upper_bound, 'g--', 'DisplayName', '+1% Error');
plot(x, lower_bound, 'g--', 'DisplayName', '-1% Error');

% Add legend, labels, and title
legend('show',Location='northwest');
xlabel('Actual L/D');
ylabel('Predicted L/D');
title('Actual vs Predicted L/D (Quadratic Regression Model');
hold off;

% Function to create polynomial features
function poly_features = createPolyFeatures(x, poly_degree)
    x_normalized = (x - mean(x)) ./ std(x);
    x_normalized(isnan(x_normalized)) = 0;
    poly_features = [];
    for i = 1:size(x_normalized, 2)
        for j = i:size(x_normalized, 2)
            poly_features = [poly_features, x_normalized(:, i) .* x_normalized(:, j)];
        end
    end
    poly_features = [x_normalized, poly_features];
end
