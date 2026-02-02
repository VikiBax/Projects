% List of files to process
file_list = {'MW-SP-Trim3.txt','MW-SP-Trim2.txt', 'Baseline4.txt'};
%file_list = {'MW-SP-Trim3-lift.txt','MW-SP-Trim2-lift.txt', 'Baseline4-lift.txt'};

% Set up the figure
figure(3);
hold on;  % Keep all plots on the same figure

for i = 1:length(file_list)
    filename = file_list{i};
    
    [x,y,ycum] = dataextract(filename);
    
    % Plot
    plot(x, ycum, '-', 'LineWidth', 1.5, 'DisplayName', filename);
end

% Finalize the plot
xlabel('X Position from Front Wheel Base');
ylabel('Accumulated Drag');
title('Accumulated Drag over X');
legend('Location', 'best');
grid on;
hold off;

%% Comparing 2 runs 

figure(4);
hold on; 
file_list = {'MW-SP-Trim2-lift.txt', 'MW-SP-Trim3-lift.txt'};

[x1,y1,y1cum] = dataextract(file_list{1});
[x2,y2,y2cum] = dataextract(file_list{2});


y = y2-y1; 
y_cs = -1*(y2cum - y1cum);

plot(x1, y_cs, '-', 'DisplayName', [file_list{1} ' - ' file_list{2}]);
xline(0.25, '--', 'DisplayName', 'SPMW start');
legend('Location', 'best');


function [x,y,ycum] = dataextract(filename)
   
    % Load the data
    data = load(filename);
    
    % Extract x and y
    x = data(:,1);
    y = data(:,2);

    %cumy
    ycum = cumsum(y);
end
