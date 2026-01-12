% Load data from CSV file
data = readmatrix('TestData.csv');

% Extract inputs (columns 4-9) and outputs (column 12)
inputs = data(:, 4:9); % Second AoA, Gap Size, etc.
outputs = data(:, 12); % L/D values

% Handle missing data by imputing missing values with column means

% Define ranges for optimization
ranges = [
    14, 22;    % Second AoA
    15, 35;    % Second Gap Size
    115, 135;  % Second Gap Angle
    30, 40;    % Third AoA
    5, 12;     % Third Gap Size
    57, 72     % Third Gap Angle
];
lb = ranges(:, 1); % Lower bounds
ub = ranges(:, 2); % Upper bounds

% **First Regression Model: Independent Variables**
disp('Running first regression model with independent variables...');
% Normalize inputs
inputs_normalized = (inputs - mean(inputs)) ./ std(inputs);

% Generate polynomial features for the regression model
poly_degree = 2;
inputs_poly = [];
for i = 1:size(inputs_normalized, 2)
    for j = i:size(inputs_normalized, 2)
        inputs_poly = [inputs_poly, inputs_normalized(:, i) .* inputs_normalized(:, j)];
    end
end
inputs_poly = [inputs_normalized, inputs_poly]; % Include original inputs

% Fit the regression model
model_1 = fitlm(inputs_poly, outputs);

% Predict outputs for all test cases
predicted_outputs_1 = predict(model_1, inputs_poly);

% Generate a new test case with optimal L/D (maximize the model prediction)
disp('Finding optimal test case for independent variables...');
initial_point = mean(ranges, 2)'; % Start at the midpoint of the ranges
optimal_inputs_1 = fmincon(@(x) -predict(model_1, createPolyFeatures(x, poly_degree)), ...
    initial_point, [], [], [], [], lb, ub);
disp('Optimal test case (independent variables):');
disp(optimal_inputs_1);

% Predict L/D ratio for the optimal test case
optimal_inputs_normalized = (optimal_inputs_1 - mean(inputs)) ./ std(inputs);
optimal_inputs_poly = createPolyFeatures(optimal_inputs_normalized, poly_degree);
predicted_LD_optimal = predict(model_1, optimal_inputs_poly);
disp(['Predicted L/D ratio for optimal inputs: ', num2str(predicted_LD_optimal)]);

% Visualization: Actual vs Predicted L/D with ±1% error bounds
figure;

% Scatter plot of actual vs predicted values
scatter(outputs, predicted_outputs_1, 'b', 'DisplayName', 'Model 1: Independent Variables');
hold on;

% Add x = y line (perfect prediction)
x = min(outputs):0.01:max(outputs); % Range of x
plot(x, x, 'k--', 'DisplayName', 'x = y'); % Perfect prediction line

% Add ±1% error bounds
upper_bound = x * 1.01; % +1% line
lower_bound = x * 0.99; % -1% line
plot(x, upper_bound, 'g--', 'DisplayName', '+1% Error');
plot(x, lower_bound, 'g--', 'DisplayName', '-1% Error');

% Add legend, labels, and title
legend('show',Location='northwest');
xlabel('Actual L/D');
ylabel('Predicted L/D');
title('Actual vs Predicted L/D with ±1% Error Bounds');
hold off;

