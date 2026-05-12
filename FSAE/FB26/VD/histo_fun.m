% 1. Generate sample data
data1 = randn(500,1);
data2 = randn(500,1) + 2;

% 2. Define common bin edges
edges = -5:0.5:7;
centers = edges(1:end-1) + diff(edges)/2;

% 3. Calculate counts for each set
counts1 = histcounts(data1, edges);
counts2 = histcounts(data2, edges);

% 4. Combine into a matrix and plot
h = bar(centers, [counts1; counts2]', 'stacked');

% 5. Apply different colors
h(1).FaceColor = [0 0.447 0.741]; % Blue
h(2).FaceColor = [0.85 0.325 0.098]; % Orange

legend('Dataset 1', 'Dataset 2');
xlabel('Bins'); ylabel('Counts');
