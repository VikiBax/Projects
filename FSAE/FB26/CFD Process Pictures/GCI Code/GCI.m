% Simple Mesh Convergence Plots for FSAE CFD Results

% Mesh data
cell_counts = [25439456, 17630561, 11759667]; % Fine, Medium, Coarse
inv_h = cell_counts.^(1/3);  % Use as mesh resolution proxy (N^{1/3})

% Coefficients
cd_car = [0.571, 0.571, 0.570];
cl_car = [1.16, 1.18, 1.16];
cl_fw  = [0.467, 0.468, 0.455];
cl_mw  = [0.0677, 0.0847, 0.0868];
cl_rw  = [0.627, 0.613, 0.612];

% --- Plot Cd (Car)
figure;
plot(inv_h, cd_car, 'o-k', 'MarkerFaceColor','k', 'LineWidth', 1.5);
xlabel('N^{1/3} (Mesh Resolution Proxy)');
ylabel('Cd (Car)');
title('Drag Coefficient Convergence');
grid on;

% --- Plot Cl (Car)
figure;
plot(inv_h, cl_car, 'o-b', 'MarkerFaceColor','b', 'LineWidth', 1.5);
xlabel('N^{1/3} (Mesh Resolution Proxy)');
ylabel('Cl (Car)');
title('Lift Coefficient Convergence: Car');
grid on;

% --- Plot Cl (Front Wing)
figure;
plot(inv_h, cl_fw, 'o-m', 'MarkerFaceColor','m', 'LineWidth', 1.5);
xlabel('N^{1/3} (Mesh Resolution Proxy)');
ylabel('Cl (Front Wing)');
title('Lift Coefficient Convergence: Front Wing');
grid on;

% --- Plot Cl (Mid Wing)
figure;
plot(inv_h, cl_mw, 'o-r', 'MarkerFaceColor','r', 'LineWidth', 1.5);
xlabel('N^{1/3} (Mesh Resolution Proxy)');
ylabel('Cl (Mid Wing)');
title('Lift Coefficient Convergence: Mid Wing');
grid on;

% --- Plot Cl (Rear Wing)
figure;
plot(inv_h, cl_rw, 'o-g', 'MarkerFaceColor','g', 'LineWidth', 1.5);
xlabel('N^{1/3} (Mesh Resolution Proxy)');
ylabel('Cl (Rear Wing)');
title('Lift Coefficient Convergence: Rear Wing');
grid on;
