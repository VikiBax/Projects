%% Optimization Setup (Gradient-based)

% Objective: minimize negative prediction => maximize prediction
obj = @(x) objective_neg_predict_with_grad(x, model, input_means, input_stds);

% Build a baseline fmincon problem
x0 = mean([lb; ub], 1);  % center of box
problem = createOptimProblem('fmincon', ...
    'objective', obj, ...
    'x0', x0, ...
    'lb', lb, ...
    'ub', ub, ...
    'options', opts);

ms = MultiStart('UseParallel', false, 'Display', 'off');
startPoints = RandomStartPointSet('NumStartPoints', nStarts);

[xbest, fbest, exitflag_best, output_best, solutions] = run(ms, problem, startPoints);

%% Collect and rank solutions (convert fbest -> predicted)
numS = numel(solutions);
X = zeros(numS, nvar);
Y = zeros(numS, 1);

for k = 1:numS
    X(k,:) = solutions(k).X(:).';
    % Predict using SAME polyfeatures
    phi = polyfeatures_quad(X(k,:), input_means, input_stds);
    Y(k) = predict(model, phi);
end

% Sort by highest predicted value
[YS, idx] = sort(Y, 'descend');
XS = X(idx,:);

% Keep only unique maxima (simple clustering by Euclidean distance)
Xuniq = [];
Yuniq = [];

for k = 1:size(XS,1)
    if isempty(Xuniq)
        Xuniq = XS(k,:);
        Yuniq = YS(k);
    else
        d = vecnorm(Xuniq - XS(k,:), 2, 2);
        if all(d > uniqueTol)
            Xuniq = [Xuniq; XS(k,:)]; 
            Yuniq = [Yuniq; YS(k)]; 
        end
    end
    if size(Xuniq,1) >= maxUnique
        break;
    end
end

%% Display results
disp('==============================================')
disp('Best local maximum found by MultiStart + fmincon:')
disp('Optimal inputs (xbest):')
disp(xbest)
disp(['Predicted value at xbest: ', num2str(-fbest)])
disp(['exitflag: ', num2str(exitflag_best)])
disp('==============================================')

disp('Top UNIQUE local maxima (ranked):')
T = table((1:size(Xuniq,1))', Yuniq, 'VariableNames', {'Rank','PredictedValue'});
disp(T)

disp('Inputs for each ranked local maximum (rows correspond to Rank):')
disp(Xuniq)

%% (Optional) verify best point prediction directly
phi_best = polyfeatures_quad(xbest, input_means, input_stds);
y_best   = predict(model, phi_best);
disp(['Sanity check predict(model, phi_best) = ', num2str(y_best)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f, g] = objective_neg_predict_with_grad(x, model, mu, sigma)
% Objective for fmincon:
%   f(x) = -prediction(x)
%   g(x) = gradient of f wrt x (analytic)

    x = x(:)'; % row

    % Feature vector and Jacobian dphi/dx
    [phi, dphi_dx] = polyfeatures_quad_row_with_grad(x, mu, sigma);

    % Linear model prediction y = Bias + phi * Beta
    beta = model.Beta;   % p x 1
    bias = model.Bias;   % scalar

    y = bias + phi * beta;      % scalar

    % dy/dx = (dphi/dx) * beta   where dphi/dx is (n x p)
    dydx = dphi_dx * beta;      % n x 1

    % Minimize negative of prediction
    f = -y;
    g = -dydx;                  % n x 1
end
