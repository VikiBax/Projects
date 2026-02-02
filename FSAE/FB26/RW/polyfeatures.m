function [poly_features] = polyfeatures(x, means, stds)
    x_normalized = (x - means) ./ stds;
    x_normalized(isnan(x_normalized)) = 0;
    poly_features = [];
    for i = 1:size(x_normalized, 2)
        for j = i:size(x_normalized, 2)
            poly_features = [poly_features, x_normalized(:, i) .* x_normalized(:, j)];
        end
    end
    poly_features = [x_normalized, poly_features];
end