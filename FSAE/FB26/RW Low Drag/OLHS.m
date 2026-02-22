% Optimized Latin Hypercube Sampling script
% User inputs
nSamples = 45;

% Midwing Configs RW AoA 2, RW AoA 3, Move 2nd X (mm), Move 2nd Z (mm)

baseline = [3, 10, 59.48, 141.54];

ub = [10, 15, 84.48, 211.54];
lb = [-5, 5, 9.48, 71.54];

nVars = length(ub);

rng('default'); 
unitLHS = lhsdesign(nSamples, nVars, 'criterion', 'maximin', 'iterations', 1000);

% Scale unit hypercube samples to [lb, ub]
lb = lb(:)'; ub = ub(:)';
scaledSamples = bsxfun(@plus, lb, bsxfun(@times, unitLHS, (ub - lb)));
scaledSamples = round(scaledSamples, 1);

% Save scaled samples to CSV file (simple)
filename = 'OLHS.csv';
writematrix(scaledSamples, filename);
