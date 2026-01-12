function poly_features = createPolyFeatures(x, poly_degree)
    % Normalize input
    x_normalized = (x - mean(x)) ./ std(x);
    x_normalized(isnan(x_normalized)) = 0; % Handle divide-by-zero cases
    
    % Generate polynomial features
    poly_features = [];
    for i = 1:size(x_normalized, 2)
        for j = i:size(x_normalized, 2)
            poly_features = [poly_features, x_normalized(:, i) .* x_normalized(:, j)];
        end
    end
    poly_features = [x_normalized, poly_features]; % Include original inputs
end
