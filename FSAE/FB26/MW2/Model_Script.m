% Main Script for Optimization Scheme 
clear; clc; 

%% Settings 
csv_name = "01_XX.csv";
row_count = 10; 
Component = "Car"; % options are "Car", "MW", "RW"

boundtype = "Data"; % Options are "Data" and "User"
ub = []; % User set ub, overriden if type set to Data
lb = []; % User set lb, overriden if type set to Data


%% Import 
raw_data = readmatrix(raw_csv);
raw_data = raw_data(1:row_count); 

% Inputs: AoA, Move X, Move Z 
inputs_raw = raw_data(:, 6:8); 

% Outputs: Lift, Drag, L/D 
switch Component 
    case "Car"
        outputs = raw_data(:,10:12);
    case "MW"
        outputs = raw_data(:,13:15);
    case "RW"
        outsputs = raw_Data(:,16:18);
end


%% Bounds
if boundtype == "Data" 
    ub = max(inputs_raw, [], 1);
    lb = min(inputs_raw, [], 1);
end


%% Normalizing Stats 
inputs_mean = mean(inputs_raw,1);
inputs_std = std(inputs_raw, 0, 1);

phi = polyfeatures(inputs_raw, inputs_mean, inputs_std); 

%% Regression Models
model_lift = fitrlinear(phi, output_lift, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda', 1e-3);

model_drag = fitrlinear(phi, output_drag, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda', 1e-3);

model_LD = fitrlinear(phi, output_drag, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda', 1e-3);

save("model_lift.mat", "model_lift");
save("model_drag.mat", "model_drag");
save("model_LD.mat", "model_LD");

% Saving important things to .mat 
% Save bounds to a .mat file
save("preload.mat", "lb", "ub",'inputs_mean', "inputs_std");