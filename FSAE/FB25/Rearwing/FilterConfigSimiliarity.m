% Input matrix 'data' (replace this with your matrix)
data = Opt_config;

% Define similarity threshold for each variable (column)
thresholds = ones(1,6)*1; % Adjust thresholds as needed

% Normalize each column independently
normalized_data = zeros(size(data)); % Preallocate normalized matrix
for col = 1:size(data, 2)
    col_mean = mean(data(:, col));         % Mean of the column
    col_std = std(data(:, col));           % Standard deviation of the column
    normalized_data(:, col) = (data(:, col) - col_mean) / col_std; % Normalize
end

% Number of rows in the data
numRows = size(normalized_data, 1);

% Logical array to mark rows for retention
keepRow = true(numRows, 1);

% Iterate through each row and compare to subsequent rows
for i = 1:numRows
    if keepRow(i) % Only process if the row hasn't been excluded
        for j = i+1:numRows
            % Check if all variables are within their thresholds
            differences = abs(normalized_data(i, :) - normalized_data(j, :));
            if all(differences < thresholds) % All variables are similar
                keepRow(j) = false; % Mark the row as too similar
            end
        end
    end
end

% Filter data to only keep unique configurations
filtered_data = data(keepRow, :);

% Display results
disp('Filtered Configurations:');
disp(filtered_data);
