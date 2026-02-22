%% Settings 

% Select the model
model_name = "model_lift.mat"; 
%model = "model_drag.mat";
%model = "model_ld.mat";

% Optimization Options
nStarts     = 1000;    % increase (100-500) for more local maxima discovery
maxUnique   = 15;      % how many unique local maxima to keep
uniqueTol   = 1e-6;    % distance tolerance for uniqueness in raw input space

% Options 
opts = optimoptions('fmincon', ...
    'Algorithm','interior-point', ...
    'SpecifyObjectiveGradient', true, ...
    'Display','iter', ...
    'MaxIterations', 500, ...
    'OptimalityTolerance', 1e-10, ...
    'StepTolerance', 1e-12);

%% Load Data
load("preload.mat")
model = load(model);



