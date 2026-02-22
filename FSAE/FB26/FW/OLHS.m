% Optimized Latin Hypercube Sampling script
% User inputs
nSamples = 8;
nVars = 2;

% Midwing Configs AoA, Move X (mm), Move Z (mm)

baseline = [10, 30];

ub = [18, 35];
lb = [5, 25];

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
