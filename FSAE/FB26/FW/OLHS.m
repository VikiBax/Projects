% Optimized Latin Hypercube Sampling script
% User inputs
nSamples = 12;
nVars = 2;

% Midwing Configs AoA, Move X (mm), Move Z (mm)

baseline = [10, 30];

ub = [18, 35];
lb = [5, 25];

nVars = length(ub);

rng('default'); 
unitLHS = lhsdesign(nSamples, nVars, 'criterion', 'maximin', 'iterations', 3000);

% Scale unit hypercube samples to [lb, ub]
lb = lb(:)'; ub = ub(:)';
scaledSamples = bsxfun(@plus, lb, bsxfun(@times, unitLHS, (ub - lb)));
scaledSamples = round(scaledSamples, 2);

% Save scaled samples to CSV file (simple)
filename = 'OLHS2.csv';
writematrix(scaledSamples, filename);



%% 
% Read the two CSV files
data1 = readmatrix('OLHS.csv');
data2 = readmatrix('OLHS2.csv');

x1 = data1(:,1); y1 = data1(:,2);
x2 = data2(:,1); y2 = data2(:,2);

% Create scatter plot
figure;
hold on;
scatter(x1, y1, 36, 'b', 'filled');
scatter(x2, y2, 36, 'r', 'filled');
hold off;

xlabel('Variable 1');
ylabel('Variable 2');
legend('OLHS.csv','OLHS2.csv','Location','best');
title('Comparison of OLHS (blue) and OLHS2 (red)');
grid on;