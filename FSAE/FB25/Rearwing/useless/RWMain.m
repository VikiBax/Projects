%Clearing Workspace 
clear; clc

%Loading data 
Set = "DataFull.csv";        %Rename to dataset
data = readmatrix(Set);

%Extracting inputs and outputs 
inputs = data(:,4:9); 
outputs = data(:,10:12);

% Normalize inputs
inputs = (inputs - mean(inputs)) ./ std(inputs);

%% Random Forest Tree
%hyperparamters
hyperparams = struct('NumTrees', 1000, ...
                     'MinLeafSize', 1, ...
                     'MaxNumSplits', Inf, ...
                     'NumPredictorsToSample', 5);
%random forest model for each paramter
rf_lift = rfmodel(inputs,outputs(:,1),hyperparams);
rf_drag = rfmodel(inputs,outputs(:,2),hyperparams);
rf_ld = rfmodel(inputs,outputs(:,3),hyperparams);

%residuals for evaluation 
residuals(:,1) = outputs(:,1) - oobPredict(rf_lift);
residuals(:,2) = outputs(:,2) - oobPredict(rf_drag);
residuals(:,3) = outputs(:,3) - oobPredict(rf_ld);

%Plotting residuals 
figure(1)
for i = 1:3 
    subplot(1,3,i)
    hold on
    histogram(residuals(:,i))
end
hold off 

%error to evaluation
ooberror(:,1) = oobError(rf_lift);
ooberror(:,2) = oobError(rf_drag);
ooberror(:,3) = oobError(rf_ld);

%plotting error
figure(2)
for i = 1:3
    subplot(1,3,i)
    hold on 
    plot(ooberror(:,i))
end
hold off 

%prediction vs actual 
predictions(:,1) = oobPredict(rf_lift);
predictions(:,2) = oobPredict(rf_drag);
predictions(:,3) = oobPredict(rf_ld);

%plotting 
for i = 1:3
    ScatterPlot(outputs(:,i),predictions(:,i),3+i,"plots")
end
