function [opt_config] = similarity(optimal_inputs,opt_config, threshold)
    arguments 
        optimal_inputs (1,:) double 
        opt_config (:,:) double 
        threshold (1,1) double = 1
    end

    fullset = [optimal_inputs; opt_config];
    n = size(fullset,1); 

    if n == 1
        opt_config = fullset;
        return 
    end

    normalized = zeros(size(fullset)); 

    for col = 1:size(fullset,2)
        col_mean = mean(fullset(:,col)); 
        col_std = std(fullset(:,col));
        normalized(:, col) = (fullset(:, col) - col_mean) / col_std;
    end

    for i = 1:size(fullset,1)
        differences = abs(normalized(i,:) - normalized(n,:));
        if all(differences > threshold)
            opt_config = fullset;
        end
    end