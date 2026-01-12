%clearing
clear
clc

% Define variable ranges
ranges = [
    0, 35;    % Second AoA
    8, 150;    % Second Gap Size (.05 to .3 times fwc2)
    70, 150;  % Second Gap Angle
    -2, 60;    % Third AoA
    7, 170;     % Third Gap Size (.05 to .3 times fwc3)
    70, 150     % Third Gap Angle
];

% Number of test points
numTests = 750; 

% Generate Latin Hypercube Sample
lhs = lhsdesign(numTests, size(ranges, 1), 'criterion','maximin', 'iterations', 1000);

% Scale LHS points to variable ranges
testPoints = lhs .* (ranges(:, 2)' - ranges(:, 1)') + ranges(:, 1)';

testpointscut = [];
%Boundary Check Condition Here 
for i=1:numTests 
    Test = BoundaryCheck(testPoints(i,:));
    if Test == true 
        testpointscut = [testpointscut; testPoints(i,:)];
    end
end

% Specify the starting number for file naming
startingNumber = 1; % Replace with desired starting number

% Add the starting number as the first column in the output
testpointscut = [(startingNumber:(startingNumber + size(testpointscut,1) -1 ))', testpointscut];


% Convert to table for CSV export
testTable = array2table(testpointscut, 'VariableNames', ...
    {'StartNumber', 'SecondAoA', 'SecondGapSize', 'SecondGapAngle', ...
     'ThirdAoA', 'ThirdGapSize', 'ThirdGapAngle'});

% Save to a CSV file
writetable(testTable, 'TestPoints.csv');

disp('Test cases with starting numbers saved to TestPoints.csv');

%Seeing the distribution 
figure(1)
subplot(1,2,1)
scatter3(testpointscut(:,2),testpointscut(:,3),testpointscut(:,4))
xlabel("Second AoA") ; ylabel("Second Gap Size") ; zlabel("Second Gap Angle");
axis([ranges(1,:)  ranges(2,:) ranges(3,:)])
subplot(1,2,2)
scatter3(testpointscut(:,5),testpointscut(:,6),testpointscut(:,7))
xlabel("Third AoA") ; ylabel("Third Gap Size") ; zlabel("Third Gap Angle");
axis([ranges(4,:)  ranges(5,:) ranges(6,:)])

figure(2)
names = {'Second AoA', 'Second Gap Size', 'Second Gap Angle', ...
    'Third AoA','Third Gap Size', 'Third Gap Angle'};

for i = 1:6
    subplot(2,3,i)
    scatter(testpointscut(:,i+1),1)
    hold on 
    title(names{i})
    ylim([0 3])
end
