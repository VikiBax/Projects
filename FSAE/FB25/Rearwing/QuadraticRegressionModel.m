%Clearing Workspace 
clear; clc

%Loading data 
Set = "DataFull.csv";        %Rename to dataset
data = readmatrix(Set);

%Extracting inputs and outputs 
inputs = data(:,4:9); 
outputs = data(:,10:12);

%inputs = inputs(1:50,:);
%outputs = outputs(1:50,:);

%% Polynomial features
%Normalize inputs 
inputs_normalized = (inputs - mean(inputs)) ./ std(inputs);

%Set degree of model (Should be locked at quadratic) 
poly_degree = 2;

inputs_poly = [];
for i = 1:size(inputs_normalized, 2)
    for j = i:size(inputs_normalized, 2)
        inputs_poly = [inputs_poly, inputs_normalized(:, i) .* inputs_normalized(:, j)];
    end
end
inputs_poly = [inputs_normalized, inputs_poly]; % Include original inputs

%% Regression Model 
% Fit quadratic regression model
model_lift = fitlm(inputs_poly, outputs(:,1));
model_drag = fitlm(inputs_poly, outputs(:,2));
model_ld = fitlm(inputs_poly, outputs(:,3));

% Evaluating model performance
rsquared_lift = model_lift.Rsquared.Adjusted;
rsquared_drag = model_drag.Rsquared.Adjusted;
rsquared_ld = model_ld.Rsquared.Adjusted;
disp(['R^2 of Lift Model: ', num2str(rsquared_lift)]);
disp(['R^2 of Drag Model: ', num2str(rsquared_drag)]);
disp(['R^2 of LD Model: ', num2str(rsquared_ld)]);

% Predict outputs for all test cases
predicted_lift = predict(model_lift, inputs_poly);
predicted_drag = predict(model_drag, inputs_poly);
predicted_ld = predict(model_ld, inputs_poly);

%Scatter Plots 
ScatterPlot(outputs(:,1),predicted_lift,1,"Lift Predctions")
ScatterPlot(outputs(:,2),predicted_drag,2,"Drag Predictions")
ScatterPlot(outputs(:,3),predicted_ld,2,"LD Predictions")


ranges = [
    min(inputs(:,1)), max(inputs(:,1));    % Second AoA
    min(inputs(:,2)), max(inputs(:,2));    % Second Gap Size (.05 to .3 times fwc2)
    min(inputs(:,3)), max(inputs(:,3));  % Second Gap Angle
    min(inputs(:,4)), max(inputs(:,4));    % Third AoA
    min(inputs(:,5)), max(inputs(:,5));     % Third Gap Size (.05 to .3 times fwc3)
    min(inputs(:,6)), max(inputs(:,6))     % Third Gap Angle
];

lb = ranges(:, 1); % Lower bounds
ub = ranges(:, 2); % Upper bounds

model = model_lift;
temp = load("ValidationModel.mat");
validmodel = temp.mdl;

% Find optimal test case using global optimization
options = optimoptions('particleswarm', ...
                       'Display', 'final', ...
                       'MaxIterations', 500 ...
                       ... %,'HybridFcn', 'fmincon' ...
                       );

ConstrainedOptimization = @(x) -predict(model, ...
        createPolyFeatures((x - mean(inputs)) ./ std(inputs), poly_degree)) ... 
        + BoundaryPenalty(x,validmodel); 
    

Opt_config = [];
for i = 1:10
    optimal_inputs_1 = particleswarm(ConstrainedOptimization, ...
        size(inputs, 2), lb, ub, options);
    Opt_config(i,:) = optimal_inputs_1; 
    display(i);
end

%disp('Optimal test case using particleswarm:');
%disp(optimal_inputs_1);


%% Predict L/D for the optimal inputs
optimal_inputs_normalized = (optimal_inputs_1 - mean(inputs)) ./ std(inputs);
optimal_inputs_poly = createPolyFeatures(optimal_inputs_normalized, poly_degree);
predicted_LD_optimal = predict(model, optimal_inputs_poly);
disp(['Predicted L/D ratio for optimal inputs: ', num2str(predicted_LD_optimal)]);
