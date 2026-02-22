%% Clearing Workspace
clear; clc;

%% Import Data
raw_csv  = "01_89.csv";
raw_data = readmatrix(raw_csv);
raw_data = raw_data(1:89,:);

%% Inputs / Outputs
inputs_raw = raw_data(:, 6:13);     % AoA1..AoA3, X1..X2, Z1..Z2, offset
outputs    = raw_data(:, 14:17);    % Car Lift ; Car Drag ; RW Lift ; RW Drag

AoA1   = inputs_raw(:,1);
AoA2   = inputs_raw(:,2);
AoA3   = inputs_raw(:,3);
X1     = inputs_raw(:,4);
X2     = inputs_raw(:,5);
Z1     = inputs_raw(:,6);
Z2     = inputs_raw(:,7);
offset = inputs_raw(:,8);

output_lift = outputs(:,3);         % RW Lift
idx = (1:size(inputs_raw,1))';

%% Bounds
% breakpoints are the LAST index of each bound set (except the final set)
boundsbreakpoints = [79];  % set1: 1..79, set2: 80..end

% Order: AoA1 AoA2 AoA3 X1 X2 Z1 Z2
bU = [11.12, 30.84, 60.98,  4.04,  8.92, 22.31, 14.46;  % boundset 1
      11.12, 32.84, 62.98,  4.04,  8.92, 23.31, 15.46]; % boundset 2

bC = [ 8.20, 26.85, 57.08,  0.14,  4.97, 18.38, 10.48]; % constant (all sets)

bL = [ 5.28, 22.91, 53.13, -3.86,  1.14, 14.47,  6.51;  % boundset 1
       5.28, 22.91, 53.13, -4.86, -1.00, 13.47,  6.51]; % boundset 2


setcutoffs = [47, 57, 67, 71, 75, 78, 81, 84, 87]; 

%% Color Map (RW Lift capped)
cap_min = 140;
cap_max = 150;
cmap = parula(256);

lift_capped = min(max(output_lift, cap_min), cap_max);
lift_norm   = (lift_capped - cap_min) ./ max(eps, (cap_max - cap_min)); % [0,1]
cidx        = max(1, min(size(cmap,1), round(1 + lift_norm*(size(cmap,1)-1))));
point_colors = cmap(cidx, :);

%% Plot each variable in its own figure (no subplots)
vars  = {AoA1, AoA2, AoA3, X1, X2, Z1, Z2, offset, output_lift };
names = {'AoA1','AoA2','AoA3','X1','X2','Z1','Z2','offset', 'lift'};

for k = 1:numel(vars)
    figure(k); clf;
    scatter(idx, vars{k}, 36, point_colors, 'filled'); hold on;
    grid on;

    xlabel('Index');
    ylabel(names{k});
    title(sprintf('%s vs Index (colored by RW Lift)', names{k}));

    % Bounds only exist for first 7 vars (offset has none)
    if k <= 7
        local_add_piecewise_bounds(idx, boundsbreakpoints, bU(:,k), bC(k), bL(:,k), setcutoffs+0.5);
    end
    if k <= 8
        local_add_cbar(cmap, cap_min, cap_max);
    end

    if k == 9
        ylim([130,150])
        for xpoint = setcutoffs+0.5
            xline(xpoint, 'k:', 'LineWidth', 1);
        end
    end
    
    hold off;
end

%% ---- Local functions ----
function local_add_piecewise_bounds(idx, breakpoints, bU_sets, bC, bL_sets, xlines)
% Draw piecewise-constant bounds across index segments
% breakpoints = [79] means:
%   set1 over [min..79], set2 over [80..max]
    xl = [min(idx), max(idx)];

    edges = [xl(1), breakpoints(:).', xl(2)]; % segment edges in index space

    for s = 1:numel(edges)-1
        x1 = edges(s);
        x2 = edges(s+1);

        % Convert edges into inclusive-looking segments
        if s < numel(edges)-1
            x2_plot = x2;          % up to breakpoint
        else
            x2_plot = x2;          % final to end
        end

        plot([x1 x2_plot], [bU_sets(s) bU_sets(s)], 'r-', 'LineWidth', 1.25);
        plot([x1 x2_plot], [bC       bC      ],    'k-', 'LineWidth', 1.25);
        plot([x1 x2_plot], [bL_sets(s) bL_sets(s)], 'b-', 'LineWidth', 1.25); 

    
        for xpoint = xlines
            xline(xpoint, 'k:', 'LineWidth', 1);
        end
    end
end

function local_add_cbar(cmap, vmin, vmax)
    colormap(cmap);
    cb = colorbar;
    cb.Label.String = 'RW Lift (capped)';
    cb.Ticks = linspace(0,1,5);
    cb.TickLabels = arrayfun(@(v) num2str(v,'%.3g'), linspace(vmin, vmax, 5), 'UniformOutput', false);
end
