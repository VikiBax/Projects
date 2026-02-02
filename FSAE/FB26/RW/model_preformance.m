function [] = model_preformance(model,inputs_poly, real)

rsquared = model.Rsquared.Adjusted;

disp(['R^2 of Model: ', num2str(rsquared)]);

prediction = predict(model, inputs_poly);

figure
scatter(real, prediction, 5, 'filled'); hold on
title('Predictions'); xlabel('Actual'); ylabel('Predicted');
plot(real, real, 'r--');
plot(real, real * 1.01, 'k:'); % +1 upper error line
plot(real, real * 0.99, 'k:'); % -1 lower error line
hold off 

end