%% car - rw difference

%% Clearing Workspace
clear; clc;

%% Import Data
raw_csv  = "01_89.csv";
raw_data = readmatrix(raw_csv);
raw_data = raw_data(1:89,:);

%% Inputs / Outputs
% Inputs: AoA1 ; AoA2 ; AoA3 ; X1 ; X2 ; Z1 ; Z2 ; offset
inputs_raw = raw_data(:, 6:13);

% Outputs: Car Lift ; Car Drag ; RW Lift ; RW Drag
outputs = raw_data(:, 14:17);


car_lift = outputs(:,1); % Car Lift
car_drag = outputs(:,2); % Car Drag
rw_lift = outputs(:,3); % RW Lift
rw_drag = outputs(:,4); % RW Drag


rw_lift_dif = rw_lift - rw_lift(1);
rw_drag_dif = rw_drag - rw_drag(1);
car_lift_dif = car_lift - car_lift(1);
car_drag_dif = car_drag - car_drag(1);

% compute differences between RW and car
lift_dif = rw_lift_dif - car_lift_dif;
drag_dif = rw_drag_dif - car_drag_dif;

% detect outliers (use robust z-score via median and MAD)
mad_l = mad(lift_dif,1); med_l = median(lift_dif);
mad_d = mad(drag_dif,1); med_d = median(drag_dif);

% avoid division by zero
if mad_l == 0, mad_l = eps; end
if mad_d == 0, mad_d = eps; end

z_l = 0.6745*(lift_dif - med_l)./mad_l; % approx z from MAD
z_d = 0.6745*(drag_dif - med_d)./mad_d;

thresh = 5; % threshold for outlier
outlier_idx_lift = find(abs(z_l) > thresh);
outlier_idx_drag = find(abs(z_d) > thresh);

% combined outliers (either lift or drag)
outlier_idx = unique([outlier_idx_lift; outlier_idx_drag]);

% display or store results
fprintf('Outlier indices (lift): %s\n', mat2str(outlier_idx_lift'));
fprintf('Outlier indices (drag): %s\n', mat2str(outlier_idx_drag'));
fprintf('Combined outlier indices: %s\n', mat2str(outlier_idx'));

% provide vectors marking outliers for plotting if needed
is_outlier = false(size(lift_dif));
is_outlier(outlier_idx) = true;

idx = 1:1:89; 

figure(1)
scatter(idx,lift_dif); 
figure(2)
scatter(idx, drag_dif);


