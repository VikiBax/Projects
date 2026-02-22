%% Parralel Plots
clear; clc; close all;

%% Load Data
raw_csv  = "01_89.csv";
raw_data = readmatrix(raw_csv);
raw_data = raw_data(1:89,:);

% Inputs: AoA1 ; AoA2 ; AoA3 ; X1 ; X2 ; Z1 ; Z2
X = raw_data(:, 6:12);

% Output: RW Lift
y = raw_data(:, 17);  % RW Lift

valid = all(isfinite(X),2) & isfinite(y);
X = X(valid,:);
y = y(valid);

%% Normalize each column to [0,1] for visualization
xmin = min(X,[],1);
xmax = max(X,[],1);
range = xmax - xmin;
range(range == 0) = 1;

Xn = (X - xmin) ./ range;

labels = {'AoA1','AoA2','AoA3','X1','X2','Z1','Z2'};
nVar   = size(Xn,2);

%% Sort by lift so high-lift lines draw on top (optional but nice)
[ys, idx] = sort(y, 'ascend');
Xn = Xn(idx,:);
y  = ys;

%% Build continuous colormap mapping (Lift -> color index)
nC   = 256;
cmap = jet(nC);

ymin = min(y);
ymax = max(y);

% If all lifts identical, avoid divide-by-zero
if ymax == ymin
    yn = 0.5 * ones(size(y));
else
    yn = (y - ymin) / (ymax - ymin);   % 0..1
end

cidx = max(1, min(nC, round(yn*(nC-1)) + 1));  % 1..nC

%% Determine Top 10%
p90 = prctile(y, 90);          % 90th percentile threshold
isTop = y >= p90;              % logical index

%% Plot
figure('Color','w'); hold on;

% Plot lower 90% first (blue)
for i = 1:size(Xn,1)
    if ~isTop(i)
        plot(1:nVar, Xn(i,:), ...
            'Color', [0 0.3 1], ...   % blue
            'LineWidth', 0.8);
    end
end

% Plot top 10% on top (red, thicker)
for i = 1:size(Xn,1)
    if isTop(i)
        plot(1:nVar, Xn(i,:), ...
            'Color', [1 0 0], ...     % red
            'LineWidth', 2);
    end
end

% Axes styling
xlim([1 nVar]);
ylim([0 1]);
xticks(1:nVar);
xticklabels(labels);
ylabel('Normalized Parameter Value');
grid on;
title('Rear Wing Design Space (Top 10% RW Lift in Red)');

legend({'Bottom 90%','Top 10%'}, 'Location','best');

