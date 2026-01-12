function [] = ScatterPlot(actual, predicted, fignum,name )

figure(fignum)
% Scatter plot of actual vs predicted values
scatter(actual, predicted, 'b', 'DisplayName', name);
hold on;

% Add x = y line (perfect prediction)
x = min(actual):0.01:max(actual);
plot(x, x, 'k--', 'DisplayName', 'x = y');

% Add ±1% error bounds
upper_bound = x * 1.01; % +1% line
lower_bound = x * 0.99; % -1% line
plot(x, upper_bound, 'g--', 'DisplayName', '+1% Error');
plot(x, lower_bound, 'g--', 'DisplayName', '-1% Error');

% Add legend, labels, and title
legend('show',Location='northwest');
title(name);
hold off;