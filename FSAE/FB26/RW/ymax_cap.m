function [ymax_reasonable] = ymax_cap(model,x_poly, y_target)

yhat_train = predict(model, x_poly);
resid = y_target - yhat_train; 
sig_resid = std(resid);
ymax_reasonable = max(y_target) + 2*sig_resid; 

end