%% MW2 3D interpolation + visualization (RW Lift scalar field)
clear; clc; close all;

% ---- Load CSV (preserve headers exactly as in the file) ----
csvPath = "FB26 CFD Mastersheet - MW2 Tuning.csv";
T = readtable(csvPath, "VariableNamingRule","preserve");

% ---- Pull columns ----
x = T.("MW2 AoA (*)");       % MW2 AoA
y = T.("MW2 Move X (mm)");   % MW2 Move X
z = T.("MW2 Move Z (mm)");   % MW2 Move Z
v = T.("RW Lift (N)");       % Scalar to interpolate

% ---- Clean data ----
good = isfinite(x) & isfinite(y) & isfinite(z) & isfinite(v);
x = x(good); y = y(good); z = z(good); v = v(good);

% ---- Build a structured 3D query grid (MESHGRID format for slice) ----
nx = 30; ny = 30; nz = 30;

xq = linspace(min(x), max(x), nx);
yq = linspace(min(y), max(y), ny);
zq = linspace(min(z), max(z), nz);

% meshgrid ordering: Xq(i,j,k) varies with xq across columns (2nd dim)
% but meshgrid returns arrays sized [ny, nx, nz] when given (x,y,z)
[Xq, Yq, Zq] = meshgrid(xq, yq, zq);

% ---- Interpolate scattered data onto the grid ----
% griddata supports 3D scattered interpolation in modern MATLAB versions
Vq = griddata(x, y, z, v, Xq, Yq, Zq, "natural");

% Fill holes outside convex hull if desired (otherwise they stay NaN)
Vq = fillmissing(Vq, "nearest");

%% ---- Visualization 1: Slice planes ----
figure("Color","w");

xs = [min(xq) mean(xq) max(xq)];
ys = [min(yq) mean(yq) max(yq)];
zs = [min(zq) mean(zq) max(zq)];

h = slice(Xq, Yq, Zq, Vq, xs, ys, zs);
set(h, "EdgeColor","none");
shading interp; colorbar;
xlabel("MW2 AoA (*)"); ylabel("MW2 Move X (mm)"); zlabel("MW2 Move Z (mm)");
title("RW Lift (N) — Interpolated 3D Field (Slices)");
grid on; view(3);

%% ---- Visualization 2: Isosurface (constant RW Lift surface) ----
figure("Color","w");

isoVal = mean(v, "omitnan");   % choose your iso value
p = patch(isosurface(Xq, Yq, Zq, Vq, isoVal));
isonormals(Xq, Yq, Zq, Vq, p);
set(p, "FaceColor",[0.85 0.2 0.2], "EdgeColor","none");
camlight; lighting gouraud; grid on; view(3);
xlabel("MW2 AoA (*)"); ylabel("MW2 Move X (mm)"); zlabel("MW2 Move Z (mm)");
title(sprintf("Isosurface of RW Lift = %.2f N", isoVal));
