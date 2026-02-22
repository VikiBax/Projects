% Optimized Latin Hypercube Sampling script
% User inputs
nSamples = 35;
nVars = 5;

% Midwing Configs AoA, Move X (mm), Move Z (mm)

baseline = [38, 19.05, 22.86];

ub = [41, 25.4, 27.94];
lb = [34, 12.7, 21.59];

nVars = length(ub);

rng('default'); 
unitLHS = lhsdesign(nSamples, nVars, 'criterion', 'maximin', 'iterations', 1000);

% Scale unit hypercube samples to [lb, ub]
lb = lb(:)'; ub = ub(:)';
scaledSamples = bsxfun(@plus, lb, bsxfun(@times, unitLHS, (ub - lb)));
scaledSamples = round(scaledSamples, 2);

% Save scaled samples to CSV file (simple)
filename = 'OLHS.csv';
writematrix(scaledSamples, filename);
