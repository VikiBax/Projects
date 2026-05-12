clear; clc; close all;

%% ================= USER SETTINGS =================

inputFile = '2026-03-11_17.17.17.csv';

CFDVelocity_mps = 15.0;

% Heat map settings
nBins = 100;          % number of bins in each direction
useLogScale = true;   % true = log10(count+1), false = raw counts

%% ================= READ DATA =================

T = readtable(inputFile);

beta  = T.Beta;
pitch = T.Pitch;
roll  = T.Roll;
heave = T.Zcg_SM;



velocity = T.Vx;

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

%% ================= FIGURE 1: PAIRWISE HEAT MAPS =================

fig1 = figure('Color','w','Name','Figure 1 - FSAE Aero Pairwise Heat Maps');
tl1 = tiledlayout(fig1, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% 1) Beta vs Pitch
ax1 = nexttile(tl1);
plotHeatMapWithOverlay(ax1, X4(:,1), X4(:,2), nBins, useLogScale, ...
    symmetryOnly(:,1), symmetryOnly(:,2), ...
    attitudeGrid(:,1), attitudeGrid(:,2), ...
    'Beta [deg]', 'Pitch [deg]', 'Beta vs Pitch');

% 2) Beta vs Roll
ax2 = nexttile(tl1);
plotHeatMapWithOverlay(ax2, X4(:,1), X4(:,3), nBins, useLogScale, ...
    symmetryOnly(:,1), symmetryOnly(:,3), ...
    attitudeGrid(:,1), attitudeGrid(:,3), ...
    'Beta [deg]', 'Roll [deg]', 'Beta vs Roll');

% 3) Beta vs Heave
ax3 = nexttile(tl1);
plotHeatMapWithOverlay(ax3, X4(:,1), X4(:,4), nBins, useLogScale, ...
    symmetryOnly(:,1), symmetryOnly(:,4), ...
    attitudeGrid(:,1), attitudeGrid(:,4), ...
    'Beta [deg]', 'Heave [m]', 'Beta vs Heave');

% 4) Pitch vs Roll
ax4 = nexttile(tl1);
plotHeatMapWithOverlay(ax4, X4(:,2), X4(:,3), nBins, useLogScale, ...
    symmetryOnly(:,2), symmetryOnly(:,3), ...
    attitudeGrid(:,2), attitudeGrid(:,3), ...
    'Pitch [deg]', 'Roll [deg]', 'Pitch vs Roll');

% 5) Pitch vs Heave
ax5 = nexttile(tl1);
plotHeatMapWithOverlay(ax5, X4(:,2), X4(:,4), nBins, useLogScale, ...
    symmetryOnly(:,2), symmetryOnly(:,4), ...
    attitudeGrid(:,2), attitudeGrid(:,4), ...
    'Pitch [deg]', 'Heave [m]', 'Pitch vs Heave');

% 6) Roll vs Heave
ax6 = nexttile(tl1);
plotHeatMapWithOverlay(ax6, X4(:,3), X4(:,4), nBins, useLogScale, ...
    symmetryOnly(:,3), symmetryOnly(:,4), ...
    attitudeGrid(:,3), attitudeGrid(:,4), ...
    'Roll [deg]', 'Heave [m]', 'Roll vs Heave');

%% ================= FIGURE 2: VELOCITY VS OTHER VARIABLES =================

fig2 = figure('Color','w','Name','Figure 2 - Velocity Heat Maps');
tl2 = tiledlayout(fig2, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Velocity vs Beta
axV1 = nexttile(tl2);
plotHeatMapOnly(axV1, velocity, beta, nBins, useLogScale, ...
    sprintf('%s', "Velocity"), 'Beta [deg]', 'Velocity vs Beta');

% Velocity vs Pitch
axV2 = nexttile(tl2);
plotHeatMapOnly(axV2, velocity, pitch, nBins, useLogScale, ...
    sprintf('%s', "Velocity"), 'Pitch [deg]', 'Velocity vs Pitch');

% Velocity vs Roll
axV3 = nexttile(tl2);
plotHeatMapOnly(axV3, velocity, roll, nBins, useLogScale, ...
    sprintf('%s', "Velocity"), 'Roll [deg]', 'Velocity vs Roll');

% Velocity vs Heave
axV4 = nexttile(tl2);
plotHeatMapOnly(axV4, velocity, heave, nBins, useLogScale, ...
    sprintf('%s', "Velocity"), 'Heave [m]', 'Velocity vs Heave');

%% ================= 3D VISUALIZATIONS =================

% Figure 3 — Beta vs Pitch vs Roll
fig3 = figure('Color','w', 'Name','Figure 3 - Beta vs Pitch vs Roll');
ax7 = axes(fig3);

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
grid(ax7,'on');
box(ax7,'on');
view(ax7,35,25);
rotate3d(fig3,'on');

% Figure 4 — Beta vs Pitch vs Heave
fig4 = figure('Color','w', 'Name','Figure 4 - Beta vs Pitch vs Heave');
ax8 = axes(fig4);

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
grid(ax8,'on');
box(ax8,'on');
view(ax8,35,25);
rotate3d(fig4,'on');

% Figure 5 — Beta vs Roll vs Heave
fig5 = figure('Color','w', 'Name','Figure 5 - Beta vs Roll vs Heave');
ax9 = axes(fig5);

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
grid(ax9,'on');
box(ax9,'on');
view(ax9,35,25);
rotate3d(fig5,'on');

%% ================= PRINT RESULTS =================

fprintf('\nDone.\n');
fprintf('Detected velocity column: %s\n', "Velocity");
fprintf('CFD runs generated: %d\n', height(runTable));
fprintf('Symmetry-implied runs: %d\n', size(symmetryOnly,1));
fprintf('Constant CFD velocity used: %.2f m/s\n', CFDVelocity_mps);
fprintf('Total represented points: %d\n', height(runTable) + size(symmetryOnly,1));

%% ================= LOCAL FUNCTIONS =================

function plotHeatMapWithOverlay(ax, xData, yData, nBins, useLogScale, ...
    symX, symY, cfdX, cfdY, xLabelText, yLabelText, plotTitleText)

    xMin = min(xData);
    xMax = max(xData);
    yMin = min(yData);
    yMax = max(yData);

    if abs(xMax - xMin) < eps
        xMin = xMin - 0.5;
        xMax = xMax + 0.5;
    end
    if abs(yMax - yMin) < eps
        yMin = yMin - 0.5;
        yMax = yMax + 0.5;
    end

    xEdges = linspace(xMin, xMax, nBins+1);
    yEdges = linspace(yMin, yMax, nBins+1);

    N = histcounts2(xData, yData, xEdges, yEdges);

    xCenters = 0.5 * (xEdges(1:end-1) + xEdges(2:end));
    yCenters = 0.5 * (yEdges(1:end-1) + yEdges(2:end));

    if useLogScale
        Nplot = log10(N' + 1);
    else
        Nplot = N';
    end

    Nplot(N' == 0) = NaN;

    imagesc(ax, xCenters, yCenters, Nplot);
    set(ax, 'YDir', 'normal');
    hold(ax, 'on');

    colormap(ax, turbo);
    cb = colorbar(ax);
    if useLogScale
        cb.Label.String = 'log_{10}(count + 1)';
    else
        cb.Label.String = 'Count';
    end

    if ~isempty(symX)
        scatter(ax, symX, symY, 50, 'b', 'o', 'LineWidth', 1.2);
    end

    scatter(ax, cfdX, cfdY, 75, 'r', 'x', 'LineWidth', 1.6);

    xlabel(ax, xLabelText);
    ylabel(ax, yLabelText);
    title(ax, plotTitleText);
    grid(ax, 'on');
    box(ax, 'on');
end

function plotHeatMapOnly(ax, xData, yData, nBins, useLogScale, xLabelText, yLabelText, plotTitleText)

    xMin = min(xData);
    xMax = max(xData);
    yMin = min(yData);
    yMax = max(yData);

    if abs(xMax - xMin) < eps
        xMin = xMin - 0.5;
        xMax = xMax + 0.5;
    end
    if abs(yMax - yMin) < eps
        yMin = yMin - 0.5;
        yMax = yMax + 0.5;
    end

    xEdges = linspace(xMin, xMax, nBins+1);
    yEdges = linspace(yMin, yMax, nBins+1);

    N = histcounts2(xData, yData, xEdges, yEdges);

    xCenters = 0.5 * (xEdges(1:end-1) + xEdges(2:end));
    yCenters = 0.5 * (yEdges(1:end-1) + yEdges(2:end));

    if useLogScale
        Nplot = log10(N' + 1);
    else
        Nplot = N';
    end

    Nplot(N' == 0) = NaN;

    imagesc(ax, xCenters, yCenters, Nplot);
    set(ax, 'YDir', 'normal');

    colormap(ax, turbo);
    cb = colorbar(ax);
    if useLogScale
        cb.Label.String = 'log_{10}(count + 1)';
    else
        cb.Label.String = 'Count';
    end

    xlabel(ax, xLabelText);
    ylabel(ax, yLabelText);
    title(ax, plotTitleText);
    grid(ax, 'on');
    box(ax, 'on');
end