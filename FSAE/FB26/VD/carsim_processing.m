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

X4 = [beta pitch roll heave];

%% ================= STRUCTURED GRID BOUNDS =================

LP = 5;
UP = 95;

betaL_raw  = prctile(X4(:,1), LP-2);
betaH_raw  = prctile(X4(:,1), UP+2);

pitchL = prctile(X4(:,2), LP);
pitchH = prctile(X4(:,2), UP);

rollL_raw  = prctile(X4(:,3), LP);
rollH_raw  = prctile(X4(:,3), UP);

heaveL = prctile(X4(:,4), LP);
heaveH = prctile(X4(:,4), UP);

pitchC = 0;
heaveC = median(X4(:,4));

%% ================= FORCE TRUE LEFT-RIGHT SYMMETRIC MAGNITUDES =================
% Use symmetric beta/roll magnitudes so mirrored points land on the true opposite side.

betaMag = max(abs([betaL_raw, betaH_raw]));
rollMag = max(abs([rollL_raw, rollH_raw]));

betaC = 0;
rollC = 0;

betaExtreme = +betaMag;   % only solve one extreme side in CFD
rollExtreme = -rollMag;   % coupled with betaExtreme based on your data

%% ================= LEVELS =================
% CFD runs:
%   beta-roll centerline
%   one extreme coupled side only
%
% pitch/heave remain 3-level.

pitchLevels = [pitchL, pitchC, pitchH];
heaveLevels = [heaveL, heaveC, heaveH];

%% ================= BUILD CFD RUNS =================
% Centerline beta-roll cases
% One extreme coupled beta-roll side:
%   (+beta, -roll)
%
% The opposite side:
%   (-beta, +roll)
% will be created only as symmetry-implied runs.

attitudeGrid = zeros(0,4);

for j = 1:length(pitchLevels)
    for k = 1:length(heaveLevels)

        % Center beta-roll case
        attitudeGrid(end+1,:) = [ ...
            betaC, ...
            pitchLevels(j), ...
            rollC, ...
            heaveLevels(k)]; %#ok<SAGROW>

        % Extreme beta-roll case
        attitudeGrid(end+1,:) = [ ...
            betaExtreme, ...
            pitchLevels(j), ...
            rollExtreme, ...
            heaveLevels(k)]; %#ok<SAGROW>

    end
end

attitudeGrid = unique(round(attitudeGrid,10), 'rows', 'stable');

%% ================= REMOVE UNWANTED COMBINATIONS =================
% Remove:
% 1) low heave with extreme beta
% 2) high pitch with extreme beta

removeMask = false(size(attitudeGrid,1),1);

for i = 1:size(attitudeGrid,1)

    b = attitudeGrid(i,1);
    p = attitudeGrid(i,2);
    h = attitudeGrid(i,4);

    isExtremeBeta = abs(b) > 1e-12;

    lowHeaveExtremeBeta = (abs(h - heaveL) < 1e-12) && isExtremeBeta;
    highPitchExtremeBeta = (abs(p - pitchH) < 1e-12) && isExtremeBeta;

    if lowHeaveExtremeBeta || highPitchExtremeBeta
        removeMask(i) = true;
    end
end

attitudeGrid(removeMask,:) = [];
nA = size(attitudeGrid,1);

%% ================= BUILD TRUE SYMMETRY-IMPLIED RUNS =================
% Mirror only the non-center beta-roll CFD points:
%   ( beta, pitch, roll, heave ) -> ( -beta, pitch, -roll, heave )

symmetryOnly = zeros(0,4);

for i = 1:size(attitudeGrid,1)

    b = attitudeGrid(i,1);
    p = attitudeGrid(i,2);
    r = attitudeGrid(i,3);
    h = attitudeGrid(i,4);

    isCenterline = (abs(b) < 1e-12) && (abs(r) < 1e-12);

    if ~isCenterline
        mirroredPoint = [-b, p, -r, h];
        symmetryOnly(end+1,:) = mirroredPoint; %#ok<SAGROW>
    end
end

symmetryOnly = unique(round(symmetryOnly,10), 'rows', 'stable');

%% ================= BUILD CFD RUN MATRIX =================

runMat = [attitudeGrid, CFDVelocity_mps*ones(nA,1)];
caseID = strings(nA,1);

for i = 1:nA
    caseID(i) = sprintf('CASE_%03d', i);
end

runTable = array2table(runMat, ...
    'VariableNames', {'Beta_deg','Pitch_deg','Roll_deg','Heave_m','Velocity_mps'});
runTable.CaseID = caseID;
runTable = movevars(runTable, 'CaseID', 'Before', 1);

%% ================= OPTIONAL NAMED RUN TABLE =================

runTable.CaseName = strings(height(runTable),1);
for i = 1:height(runTable)
    runTable.CaseName(i) = sprintf('B_%+.2f_P_%+.2f_R_%+.2f_H_%.4f_V_%.2f', ...
        runTable.Beta_deg(i), ...
        runTable.Pitch_deg(i), ...
        runTable.Roll_deg(i), ...
        runTable.Heave_m(i), ...
        runTable.Velocity_mps(i));
end

runTable = movevars(runTable, 'CaseName', 'After', 'CaseID');

writetable(runTable, fullfile(outputFolder, 'recommended_cfd_run_matrix_filtered_symmetry.csv'));

%% ================= SAVE SYMMETRY-ONLY TABLE =================

symTable = array2table([symmetryOnly, CFDVelocity_mps*ones(size(symmetryOnly,1),1)], ...
    'VariableNames', {'Beta_deg','Pitch_deg','Roll_deg','Heave_m','Velocity_mps'});

symTable.CaseType = repmat("SymmetryOnly", height(symTable), 1);
symTable = movevars(symTable, 'CaseType', 'Before', 1);

writetable(symTable, fullfile(outputFolder, 'symmetry_implied_runs_filtered_symmetry.csv'));

%% ================= VISUALIZATIONS =================

fig = figure('Color','w','Name','FSAE Aero Pairwise Plots');
tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% 1) Beta vs Pitch
ax1 = nexttile(tl);
scatter(ax1, X4(:,1), X4(:,2), 6, [0.75 0.75 0.75], 'filled', ...
    'MarkerFaceAlpha', 0.15, 'MarkerEdgeAlpha', 0.15);
hold(ax1, 'on');
if ~isempty(symmetryOnly)
    scatter(ax1, symmetryOnly(:,1), symmetryOnly(:,2), 50, 'b', 'o', 'LineWidth', 1.2);
end
scatter(ax1, attitudeGrid(:,1), attitudeGrid(:,2), 75, 'r', 'x', 'LineWidth', 1.6);
xlabel(ax1, 'Beta [deg]');
ylabel(ax1, 'Pitch [deg]');
title(ax1, 'Beta vs Pitch');
legend(ax1, {'All lapsim data','Symmetry implied runs','CFD runs'}, 'Location', 'best');
grid(ax1, 'on');
box(ax1, 'on');

% 2) Beta vs Roll
ax2 = nexttile(tl);
scatter(ax2, X4(:,1), X4(:,3), 6, [0.75 0.75 0.75], 'filled', ...
    'MarkerFaceAlpha', 0.15, 'MarkerEdgeAlpha', 0.15);
hold(ax2, 'on');
if ~isempty(symmetryOnly)
    scatter(ax2, symmetryOnly(:,1), symmetryOnly(:,3), 50, 'b', 'o', 'LineWidth', 1.2);
end
scatter(ax2, attitudeGrid(:,1), attitudeGrid(:,3), 75, 'r', 'x', 'LineWidth', 1.6);
xlabel(ax2, 'Beta [deg]');
ylabel(ax2, 'Roll [deg]');
title(ax2, 'Beta vs Roll');
legend(ax2, {'All lapsim data','Symmetry implied runs','CFD runs'}, 'Location', 'best');
grid(ax2, 'on');
box(ax2, 'on');

% 3) Beta vs Heave
ax3 = nexttile(tl);
scatter(ax3, X4(:,1), X4(:,4), 6, [0.75 0.75 0.75], 'filled', ...
    'MarkerFaceAlpha', 0.15, 'MarkerEdgeAlpha', 0.15);
hold(ax3, 'on');
if ~isempty(symmetryOnly)
    scatter(ax3, symmetryOnly(:,1), symmetryOnly(:,4), 50, 'b', 'o', 'LineWidth', 1.2);
end
scatter(ax3, attitudeGrid(:,1), attitudeGrid(:,4), 75, 'r', 'x', 'LineWidth', 1.6);
xlabel(ax3, 'Beta [deg]');
ylabel(ax3, 'Heave [m]');
title(ax3, 'Beta vs Heave');
legend(ax3, {'All lapsim data','Symmetry implied runs','CFD runs'}, 'Location', 'best');
grid(ax3, 'on');
box(ax3, 'on');

% 4) Pitch vs Roll
ax4 = nexttile(tl);
scatter(ax4, X4(:,2), X4(:,3), 6, [0.75 0.75 0.75], 'filled', ...
    'MarkerFaceAlpha', 0.15, 'MarkerEdgeAlpha', 0.15);
hold(ax4, 'on');
if ~isempty(symmetryOnly)
    scatter(ax4, symmetryOnly(:,2), symmetryOnly(:,3), 50, 'b', 'o', 'LineWidth', 1.2);
end
scatter(ax4, attitudeGrid(:,2), attitudeGrid(:,3), 75, 'r', 'x', 'LineWidth', 1.6);
xlabel(ax4, 'Pitch [deg]');
ylabel(ax4, 'Roll [deg]');
title(ax4, 'Pitch vs Roll');
legend(ax4, {'All lapsim data','Symmetry implied runs','CFD runs'}, 'Location', 'best');
grid(ax4, 'on');
box(ax4, 'on');

% 5) Pitch vs Heave
ax5 = nexttile(tl);
scatter(ax5, X4(:,2), X4(:,4), 6, [0.75 0.75 0.75], 'filled', ...
    'MarkerFaceAlpha', 0.15, 'MarkerEdgeAlpha', 0.15);
hold(ax5, 'on');
if ~isempty(symmetryOnly)
    scatter(ax5, symmetryOnly(:,2), symmetryOnly(:,4), 50, 'b', 'o', 'LineWidth', 1.2);
end
scatter(ax5, attitudeGrid(:,2), attitudeGrid(:,4), 75, 'r', 'x', 'LineWidth', 1.6);
xlabel(ax5, 'Pitch [deg]');
ylabel(ax5, 'Heave [m]');
title(ax5, 'Pitch vs Heave');
legend(ax5, {'All lapsim data','Symmetry implied runs','CFD runs'}, 'Location', 'best');
grid(ax5, 'on');
box(ax5, 'on');

% 6) Roll vs Heave
ax6 = nexttile(tl);
scatter(ax6, X4(:,3), X4(:,4), 6, [0.75 0.75 0.75], 'filled', ...
    'MarkerFaceAlpha', 0.15, 'MarkerEdgeAlpha', 0.15);
hold(ax6, 'on');
if ~isempty(symmetryOnly)
    scatter(ax6, symmetryOnly(:,3), symmetryOnly(:,4), 50, 'b', 'o', 'LineWidth', 1.2);
end
scatter(ax6, attitudeGrid(:,3), attitudeGrid(:,4), 75, 'r', 'x', 'LineWidth', 1.6);
xlabel(ax6, 'Roll [deg]');
ylabel(ax6, 'Heave [m]');
title(ax6, 'Roll vs Heave');
legend(ax6, {'All lapsim data','Symmetry implied runs','CFD runs'}, 'Location', 'best');
grid(ax6, 'on');
box(ax6, 'on');

%% ================= 3D VISUALIZATIONS =================

% Figure 1 — Beta vs Pitch vs Roll
fig3D_1 = figure('Color','w', 'Name','Beta vs Pitch vs Roll');
ax7 = axes(fig3D_1);

scatter3(ax7, X4(:,1), X4(:,2), X4(:,3), ...
    6, [0.75 0.75 0.75], 'filled', ...
    'MarkerFaceAlpha', 0.15, 'MarkerEdgeAlpha', 0.15);
hold(ax7, 'on');

if ~isempty(symmetryOnly)
    scatter3(ax7, symmetryOnly(:,1), symmetryOnly(:,2), symmetryOnly(:,3), ...
        50, 'b', 'o', 'LineWidth', 1.2);
end

scatter3(ax7, attitudeGrid(:,1), attitudeGrid(:,2), attitudeGrid(:,3), ...
    75, 'r', 'x', 'LineWidth', 1.6);

xlabel(ax7,'Beta [deg]');
ylabel(ax7,'Pitch [deg]');
zlabel(ax7,'Roll [deg]');
title(ax7,'Beta vs Pitch vs Roll');
legend(ax7, {'All lapsim data','Symmetry implied runs','CFD runs'}, 'Location', 'best');
grid(ax7,'on');
box(ax7,'on');
view(ax7,35,25);
rotate3d(fig3D_1,'on');

% Figure 2 — Beta vs Pitch vs Heave
fig3D_2 = figure('Color','w', 'Name','Beta vs Pitch vs Heave');
ax8 = axes(fig3D_2);

scatter3(ax8, X4(:,1), X4(:,2), X4(:,4), ...
    6, [0.75 0.75 0.75], 'filled', ...
    'MarkerFaceAlpha', 0.15, 'MarkerEdgeAlpha', 0.15);
hold(ax8, 'on');

if ~isempty(symmetryOnly)
    scatter3(ax8, symmetryOnly(:,1), symmetryOnly(:,2), symmetryOnly(:,4), ...
        50, 'b', 'o', 'LineWidth', 1.2);
end

scatter3(ax8, attitudeGrid(:,1), attitudeGrid(:,2), attitudeGrid(:,4), ...
    75, 'r', 'x', 'LineWidth', 1.6);

xlabel(ax8,'Beta [deg]');
ylabel(ax8,'Pitch [deg]');
zlabel(ax8,'Heave [m]');
title(ax8,'Beta vs Pitch vs Heave');
legend(ax8, {'All lapsim data','Symmetry implied runs','CFD runs'}, 'Location', 'best');
grid(ax8,'on');
box(ax8,'on');
view(ax8,35,25);
rotate3d(fig3D_2,'on');

% Figure 3 — Beta vs Roll vs Heave
fig3D_3 = figure('Color','w', 'Name','Beta vs Roll vs Heave');
ax9 = axes(fig3D_3);

scatter3(ax9, X4(:,1), X4(:,3), X4(:,4), ...
    6, [0.75 0.75 0.75], 'filled', ...
    'MarkerFaceAlpha', 0.15, 'MarkerEdgeAlpha', 0.15);
hold(ax9, 'on');

if ~isempty(symmetryOnly)
    scatter3(ax9, symmetryOnly(:,1), symmetryOnly(:,3), symmetryOnly(:,4), ...
        50, 'b', 'o', 'LineWidth', 1.2);
end

scatter3(ax9, attitudeGrid(:,1), attitudeGrid(:,3), attitudeGrid(:,4), ...
    75, 'r', 'x', 'LineWidth', 1.6);

xlabel(ax9,'Beta [deg]');
ylabel(ax9,'Roll [deg]');
zlabel(ax9,'Heave [m]');
title(ax9,'Beta vs Roll vs Heave');
legend(ax9, {'All lapsim data','Symmetry implied runs','CFD runs'}, 'Location', 'best');
grid(ax9,'on');
box(ax9,'on');
view(ax9,35,25);
rotate3d(fig3D_3,'on');

%% ================= OPTIONAL EXPORTS =================

exportgraphics(fig, fullfile(outputFolder, 'pairwise_2D_plots_filtered_symmetry.png'), 'Resolution', 300);
exportgraphics(fig3D_1, fullfile(outputFolder, '3D_Beta_Pitch_Roll_filtered_symmetry.png'), 'Resolution', 300);
exportgraphics(fig3D_2, fullfile(outputFolder, '3D_Beta_Pitch_Heave_filtered_symmetry.png'), 'Resolution', 300);
exportgraphics(fig3D_3, fullfile(outputFolder, '3D_Beta_Roll_Heave_filtered_symmetry.png'), 'Resolution', 300);

%% ================= PRINT RESULTS =================

fprintf('\nDone.\n');
fprintf('Outputs written to folder: %s\n', outputFolder);
fprintf('CFD runs generated: %d\n', height(runTable));
fprintf('Symmetry-implied runs: %d\n', height(symTable));
fprintf('Constant CFD velocity used: %.2f m/s\n', CFDVelocity_mps);
fprintf('Total represented points: %d\n', height(runTable) + height(symTable));