%% 07_sweep_all_vars_multi_baselines.m
% Sweep each variable one-at-a-time around multiple baselines and plot predicted RW Lift
% Baselines: mean, median, best-CFD (by RW Lift)
% Outputs: one figure per variable + saved PNGs into ./Figures_Sweeps/

clear; clc; close all;

%% -------------------- Load data --------------------
raw_csv  = "01_89.csv";
raw_data = readmatrix(raw_csv);
raw_data = raw_data(1:89,:);

Xraw  = raw_data(:, 6:13);      % AoA1..AoA3, X1..X2, Z1..Z2, offset
Y     = raw_data(:, 14:17);
yLift = Y(:,3);                 % RW Lift

valid = all(isfinite(Xraw),2) & isfinite(yLift);
Xraw  = Xraw(valid,:);
yLift = yLift(valid);

N    = size(Xraw,1);
nvar = size(Xraw,2);

names = {'AoA1','AoA2','AoA3','X1','X2','Z1','Z2','offset'};

fprintf("Using N = %d samples\n", N);

%% -------------------- Bounds (use yours) --------------------
ub = [11.12, 32.84, 62.98,  4.04,  8.92, 23.31, 15.46, 10];
lb = [ 5.28, 22.91, 53.13, -4.86, -1.00, 13.47,  6.51,  5];

%% -------------------- Train surrogate (same structure) --------------------
mu  = mean(Xraw, 1);
sig = std(Xraw, 0, 1);
sig(sig==0) = 1;

Phi = polyfeatures_quad(Xraw, mu, sig);

lambda = 1e-3;
model_lift = fitrlinear(Phi, yLift, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda', lambda);

%% -------------------- Define baselines --------------------
x_mean   = mean(Xraw, 1);
x_median = median(Xraw, 1);

[~, ibest] = max(yLift);
x_best = Xraw(ibest,:);

baselines = [x_mean; x_median; x_best];
baseNames = {'Mean baseline','Median baseline','Best-CFD baseline'};

%% -------------------- Sweep settings --------------------
Nsweep = 250;            % points per sweep
useDataRangeInsteadOfBounds = false;  % true => sweep over observed min/max for each var

% Output folder
outdir = "Figures_Sweeps";
if ~exist(outdir, "dir"), mkdir(outdir); end

%% -------------------- Generate plots (one per variable) --------------------
for v = 1:nvar
    if useDataRangeInsteadOfBounds
        vmin = min(Xraw(:,v));
        vmax = max(Xraw(:,v));
    else
        vmin = lb(v);
        vmax = ub(v);
    end

    xs = linspace(vmin, vmax, Nsweep)';

    % Precompute rug positions (actual sampled values)
    xSamples = Xraw(:,v);

    figure('Color','w'); hold on; grid on;

    % Plot each baseline curve
    for b = 1:size(baselines,1)
        Xtest = repmat(baselines(b,:), Nsweep, 1);
        Xtest(:,v) = xs;

        Phi_test = polyfeatures_quad(Xtest, mu, sig);
        yhat = predict(model_lift, Phi_test);

        plot(xs, yhat, 'LineWidth', 2, 'DisplayName', baseNames{b});

        % Mark baseline value for that curve
        xline(baselines(b,v), '--', 'LineWidth', 1.0, 'HandleVisibility','off');
    end

    % Rug plot for data support (small ticks at bottom)
    yl = ylim;
    tickH = 0.03*(yl(2)-yl(1));
    y0 = yl(1);
    for i = 1:numel(xSamples)
        plot([xSamples(i) xSamples(i)], [y0 y0+tickH], 'k-', 'HandleVisibility','off');
    end

    % Labels/titles
    xlabel(names{v});
    ylabel('Predicted RW Lift (N)');
    title(sprintf('1D Surrogate Sweep: %s (others fixed at baseline)', names{v}));

    legend('Location','best');

    % Save
    fname = fullfile(outdir, sprintf("Sweep_%02d_%s.png", v, names{v}));
    exportgraphics(gcf, fname, 'Resolution', 300);
end

disp("Done. Saved plots to: " + outdir);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local functions: Quadratic feature map (matches your ordering)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function PHI = polyfeatures_quad(X, mu, sigma)
    if isvector(X)
        x = X(:)';
        PHI = polyfeatures_quad_row(x, mu, sigma);
        return;
    end
    N = size(X,1);
    n = size(X,2);
    p = 1 + 2*n + n*(n-1)/2;

    PHI = zeros(N, p);
    for r = 1:N
        PHI(r,:) = polyfeatures_quad_row(X(r,:), mu, sigma);
    end
end

function phi = polyfeatures_quad_row(x, mu, sigma)
    x     = x(:)'; 
    mu    = mu(:)'; 
    sigma = sigma(:)';

    z = (x - mu) ./ sigma;

    n = numel(z);
    nCross = n*(n-1)/2;
    p = 1 + n + n + nCross;

    phi = zeros(1,p);
    idx = 1;

    % bias
    phi(idx) = 1; idx = idx + 1;

    % linear
    for i = 1:n
        phi(idx) = z(i);
        idx = idx + 1;
    end

    % squared
    for i = 1:n
        phi(idx) = z(i)^2;
        idx = idx + 1;
    end

    % cross
    for i = 1:n-1
        for j = i+1:n
            phi(idx) = z(i)*z(j);
            idx = idx + 1;
        end
    end
end
