% Load data from CSV file
data = readmatrix('TestData.csv');

% Extract inputs (columns 4-9) and outputs (column 12)
inputs = data(:, 4:9); % Second AoA, Gap Size, etc.
outputs = data(:, 12); % L/D values

% Handle missing data by imputing missing values with column means
for col = 1:size(inputs, 2)
    nan_indices = isnan(inputs(:, col));
    inputs(nan_indices, col) = mean(inputs(~nan_indices, col));
end

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
initial_point = mean(inputs, 1); % Use mean of inputs as initial point
optimal_inputs_1 = fmincon(@(x) -predict(model_1, createPolyFeatures(x, poly_degree)), ...
    initial_point, [], [], [], [], -3 * ones(1, size(inputs, 2)), 3 * ones(1, size(inputs, 2)));
disp('Optimal test case (independent variables):');
disp(optimal_inputs_1);

% **Second Regression Model: Coupled Variables**
disp('Running second regression model with coupled variables...');
% Couple specific columns
coupled_inputs = [inputs(:, 1) ./ inputs(:, 4), inputs(:, 2) ./ inputs(:, 5)];
% Combine coupled variables with original inputs
inputs_coupled = [inputs(:, 1:3), coupled_inputs, inputs(:, 6)];

% Normalize coupled inputs
inputs_coupled_normalized = (inputs_coupled - mean(inputs_coupled)) ./ std(inputs_coupled);

% Generate polynomial features for coupled inputs
inputs_coupled_poly = [];
for i = 1:size(inputs_coupled_normalized, 2)
    for j = i:size(inputs_coupled_normalized, 2)
        inputs_coupled_poly = [inputs_coupled_poly, ...
            inputs_coupled_normalized(:, i) .* inputs_coupled_normalized(:, j)];
    end
end
inputs_coupled_poly = [inputs_coupled_normalized, inputs_coupled_poly];

% Fit the regression model for coupled inputs
model_2 = fitlm(inputs_coupled_poly, outputs);

% Predict outputs for all test cases
predicted_outputs_2 = predict(model_2, inputs_coupled_poly);

% Generate a new test case with optimal L/D for coupled variables
disp('Finding optimal test case for coupled variables...');
initial_point_coupled = mean(inputs_coupled, 1); % Use mean of coupled inputs as initial point
optimal_inputs_2 = fmincon(@(x) -predict(model_2, createPolyFeatures(x, poly_degree)), ...
    initial_point_coupled, [], [], [], [], -3 * ones(1, size(inputs_coupled, 2)), 3 * ones(1, size(inputs_coupled, 2)));
disp('Optimal test case (coupled variables):');
disp(optimal_inputs_2);

% Visualization: Actual vs Predicted for both models
figure;
scatter(outputs, predicted_outputs_1, 'b', 'DisplayName', 'Model 1: Independent Variables');
hold on;
scatter(outputs, predicted_outputs_2, 'r', 'DisplayName', 'Model 2: Coupled Variables');
x = min(outputs):0.01:max(outputs);
plot(x, x, 'k--', 'DisplayName', 'x = y'); % Perfect prediction line
legend('show');
xlabel('Actual L/D');
ylabel('Predicted L/D');
title('Actual vs Predicted L/D for Both Models');


% Add ±1% error bounds
upper_bound = x * 1.01; % +1% line
lower_bound = x * 0.99; % -1% line
plot(x, upper_bound, 'g--', 'DisplayName', '+1% Error');
plot(x, lower_bound, 'g--', 'DisplayName', '-1% Error');


hold off;

% Function to create polynomial features for new inputs
function poly_features = createPolyFeatures(x, poly_degree)
    % Normalize input
    x_normalized = (x - mean(x)) ./ std(x); % Replace with zeros if std is zero
    x_normalized(isnan(x_normalized)) = 0; % Handle divide-by-zero cases
    
    % Generate polynomial features
    poly_features = [];
    for i = 1:size(x_normalized, 2)
        for j = i:size(x_normalized, 2)
            poly_features = [poly_features, x_normalized(:, i) .* x_normalized(:, j)];
        end
    end
    poly_features = [x_normalized, poly_features]; % Include original inputs
end
