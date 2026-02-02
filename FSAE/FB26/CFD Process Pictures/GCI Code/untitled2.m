cell_counts = [11759667, 17630561, 25439456]; % Coarse, Medium, Fine
drag_vals = [78.13486, 78.68119, 78.92598];   % N
lift_vals = [161.8000, 163.4102, 159.6234];   % N


% Drag Plot
subplot(1,2,1);
loglog(cell_counts, drag_vals); hold on;


% Lift Plot
subplot(1,2,2);
loglog(cell_counts, lift_vals); hold on;