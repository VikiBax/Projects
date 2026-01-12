%Clearing 
clear; clc;

% Load data from CSV file for training
data1 = readmatrix('TestData.csv');
data2 = readmatrix('TestData2.csv');

% Extract inputs (columns 4-9) and outputs (column 12) for training
inputs1 = data1(2:end, 4:9);
inputs2 = data2(2:end, 4:9);

y1 = zeros([1,size(inputs1,1)])+1;
y2 = zeros([1,size(inputs2,1)])+2;

names = {'Second AoA', 'Second Gap Size', 'Second Gap Angle', ...
    'Third AoA','Third Gap Size', 'Third Gap Angle'};

for i = 1:6
    subplot(2,3,i)
    scatter(inputs1(:,i)',y1)
    hold on 
    scatter(inputs2(:,i)',y2)
    legend('First Data Set','Second Data Set')
    title(names{i})
    ylim([0 3])

end