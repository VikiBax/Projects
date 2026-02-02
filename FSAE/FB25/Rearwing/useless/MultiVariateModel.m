%Clearing Workspace 
clear; clc

%Loading data 
Set = "DataFull.csv";        %Rename to dataset
data = readmatrix(Set);

%Extracting inputs and outputs 
inputs = data(:,4:9); 
outputs = data(:,10:12);

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

%% multivariate model 
% adding intercept term to avoid origin 
inputs_mvreg = [ones(size(inputs,1),1), inputs_normalized];

%multivariate regression 
[beta, Sigma] = mvregress(inputs_mvreg, outputs);

%testing 
prediction = inputs_mvreg * beta; 
ScatterPlot(outputs(:,1),prediction(:,1),3,"Lift Predctions")
ScatterPlot(outputs(:,2),prediction(:,2),4,"Drag Predictions")