%clearing
clear
clc

% Define variable ranges
ranges = [
    0, 25;    % Second AoA
    6.5, 39;    % Second Gap Size (.05 to .3 times fwc2)
    10, 100;  % Second Gap Angle
    45, 75;    % Third AoA
    3.2, 19.2;     % Third Gap Size (.05 to .3 times fwc3)
    10, 100     % Third Gap Angle
];

% Number of test points
numTests = 800; % Adjust as needed

% Generate Latin Hypercube Sample
lhs = lhsdesign(numTests, size(ranges, 1));

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
startingNumber = 70; % Replace with desired starting number

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
axis([0 25 6.5 39 10 100])
subplot(1,2,2)
scatter3(testpointscut(:,5),testpointscut(:,6),testpointscut(:,7))
xlabel("Third AoA") ; ylabel("Third Gap Size") ; zlabel("Third Gap Angle");
axis([45 75 3.2 39 10 100])
