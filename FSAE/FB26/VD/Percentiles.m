clear; clc; close all;

%% ================= USER SETTINGS =================

inputFile = '2026-03-11_17.17.17.csv';
outputFolder = 'aero_map_outputs_matlab';

CFDVelocity_mps = 15.0;

%% ================= SETUP =================

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

%% ================= READ DATA =================

T = readtable(inputFile);

beta  = T.Beta;
pitch = T.Pitch;
roll  = T.Roll;
heave = T.Zcg_SM;
vel   = T.Vx/1.609; 

X4 = [beta pitch roll heave];

%% 

% create histograms for vel, beta, pitch, roll, heave
vars = {'vel','beta','pitch','roll','heave'};
data = {vel, beta, pitch, roll, heave};
colors = {[0.2 0.6 0.8], [0.8 0.4 0.4], [0.4 0.8 0.4], [0.6 0.4 0.8], [0.9 0.7 0.2]};

n = numel(data);
cols = min(3,n);
rows = ceil(n/cols);
figure('Name','Histograms','NumberTitle','off','Units','normalized','Position',[0.1 0.1 0.8 0.7]);
for i = 1:n
    subplot(rows,cols,i);
    histogram(data{i}, 'BinMethod','sturges', 'FaceColor',colors{i}, 'EdgeColor','k');
    xlabel(vars{i});
    ylabel('Count');
    title(['Histogram of ' vars{i}]);
    grid on;
end

% compute and display percentiles 10,25,50,75,99 for each variable
prct = [10 25 50 75 99];
for i = 1:n
    p = prctile(data{i}, prct);
    fprintf('%s percentiles: %d%%=%.3f, %d%%=%.3f, %d%%=%.3f, %d%%=%.3f, %d%%=%.3f\n',...
        vars{i}, prct(1), p(1), prct(2), p(2), prct(3), p(3), prct(4), p(4), prct(5), p(5));
end