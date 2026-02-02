function [] = kfold(x_poly, y_target, k, lambda)

cvp = cvpartition(size(x_poly,1), 'KFold', k);

yhat_cv = nan(size(y_target)); 

for fold = 1:k 
    tr = training(cvp, fold);
    te = test(cvp, fold);

    mdl = fitrlinear(x_poly(tr,:), y_target(tr), ...
        'Learner','leastsquares', ...
        'Regularization','ridge', ...
        'Lambda', lambda);

    yhat_cv(te) = predict(mdl, x_poly(te,:));
end

rmse = sqrt(mean((yhat_cv - y_target).^2));
mae  = mean(abs(yhat_cv - y_target));
yrng = max(y_target) - min(y_target);

disp('--- CV Diagnostics (Target) ---');
disp(['k-fold k = ', num2str(k)]);
disp(['Lambda   = ', num2str(lambda)]);
disp(['RMSE     = ', num2str(rmse)]);    % Noise 
disp(['MAE      = ', num2str(mae)]);     % Incorrectness 
disp(['Range    = ', num2str(min(y_target)), ' to ', num2str(max(y_target)), ...
      ' (range=', num2str(yrng), ')']);
disp(['RMSE / range = ', num2str(rmse / max(yrng, eps))]);

end