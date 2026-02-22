%% ramjet_calcs.m
% Ramjet performance model (quasi-1D, calorically perfect gas)
% Stations: a (freestream) -> 2 (diffuser exit / combustor inlet)
%          -> 4 (combustor exit / nozzle inlet) -> 6 (nozzle exit)
%
% Assumptions:
% - ISA sea level by default (overrideable)
% - p6 ~ pa (ideally expanded), pressure thrust neglected
% - Momentum thrust: T ≈ mdot_a * [(1+f)*V6 - Va]
% - mdot_a ≈ rho_a * Va * A_cap, with A_cap = pi d^2/4
%
% Two usage modes:
%   (A) Provide inlet diameter d [m] -> computes thrust
%   (B) Provide target thrust T_target [N] -> computes required d [m]
%
% Author: (you)
% -------------------------------------------------------------

clear; clc;

%% -------------------- User Inputs ---------------------------
% Freestream / ambient
M        = 2.0;        % Freestream Mach number [-]
Ta       = 288.15;     % Ambient static temperature [K] (ISA SL = 288.15)
pa       = 101325;     % Ambient static pressure [Pa] (ISA SL = 101325)

% Gas properties
gamma    = 1.4;        % Ratio of specific heats [-]
cp       = 1004.5;     % Specific heat at constant pressure [J/(kg*K)]
R        = cp*(1 - 1/gamma); % Gas constant [J/(kg*K)] (consistent with cp,gamma)

% Loss / efficiency models
pi_d     = 0.92;       % Diffuser total pressure recovery p0,2 / p0,a [-]
pi_b     = 0.95;       % Combustor total pressure ratio p0,4 / p0,2 [-] (must be < 1)
eta_b    = 0.98;       % Combustion efficiency [-]
eta_n    = 0.95;       % Nozzle efficiency [-]

% Fuel properties
h_pr     = 43e6;       % Fuel lower heating value [J/kg]
f        = 0.03;       % Fuel-air ratio (mdot_f / mdot_a) [-]

% Choose one mode:
mode = "given_d";      % "given_d" or "size_d_for_thrust"

% Mode A: provide inlet capture diameter
d_inlet  = 0.05;       % Inlet capture diameter [m]

% Mode B: provide target thrust to size inlet diameter
T_target = 50;         % Target net thrust [N]
% ------------------------------------------------------------

%% -------------------- Core Calculations ---------------------
out = ramjet_model(M, Ta, pa, gamma, cp, R, ...
                   pi_d, pi_b, eta_b, eta_n, ...
                   h_pr, f);

% Print core freestream + cycle info
fprintf("=== Ramjet Model Summary ===\n");
fprintf("M = %.3f\n", M);
fprintf("Va = %.3f m/s\n", out.Va);
fprintf("a  = %.3f m/s\n", out.a);
fprintf("rho_a = %.4f kg/m^3\n", out.rho_a);
fprintf("T0,a = %.2f K,  p0,a = %.1f Pa\n", out.T0a, out.p0a);
fprintf("T0,2 = %.2f K,  p0,2 = %.1f Pa\n", out.T02, out.p02);
fprintf("T0,4 = %.2f K,  p0,4 = %.1f Pa\n", out.T04, out.p04);
fprintf("Nozzle: T6,s = %.2f K, V6 = %.2f m/s\n", out.T6s, out.V6);

switch mode
    case "given_d"
        perf = ramjet_with_inlet(out, d_inlet);
        fprintf("\n--- Mode: Given inlet diameter ---\n");
        fprintf("d_inlet = %.4f m (A_cap = %.6f m^2)\n", d_inlet, perf.Acap);
        fprintf("mdot_a  = %.4f kg/s\n", perf.mdot_a);
        fprintf("Thrust  = %.3f N\n", perf.T);
        fprintf("mdot_f  = %.6f kg/s\n", perf.mdot_f);

    case "size_d_for_thrust"
        d_req = ramjet_size_inlet_for_thrust(out, T_target);
        perf  = ramjet_with_inlet(out, d_req);
        fprintf("\n--- Mode: Size inlet diameter for target thrust ---\n");
        fprintf("T_target = %.3f N\n", T_target);
        fprintf("d_req    = %.4f m\n", d_req);
        fprintf("Check: Thrust = %.3f N\n", perf.T);
        fprintf("mdot_a   = %.4f kg/s, mdot_f = %.6f kg/s\n", perf.mdot_a, perf.mdot_f);

    otherwise
        error("Unknown mode. Use 'given_d' or 'size_d_for_thrust'.");
end

%% -------------------- Local Functions -----------------------
function out = ramjet_model(M, Ta, pa, gamma, cp, R, pi_d, pi_b, eta_b, eta_n, h_pr, f)
% Computes thermodynamic / nozzle quantities independent of inlet size.

    % Freestream
    a   = sqrt(gamma*R*Ta);
    Va  = M*a;
    rho_a = pa/(R*Ta);

    % Stagnation properties (a -> 1, with 1 identical to a here)
    T0a = Ta*(1 + 0.5*(gamma-1)*M^2);
    p0a = pa*(1 + 0.5*(gamma-1)*M^2)^(gamma/(gamma-1));

    % Diffuser (1 -> 2): T0 constant, p0 recovered by pi_d
    T02 = T0a;
    p02 = pi_d*p0a;

    % Combustor (2 -> 4)
    % Energy balance: (1+f)cp*T04 = cp*T02 + eta_b*f*h_pr
    % => T04 = (T02 + eta_b*f*h_pr/cp)/(1+f)
    T04 = (T02 + eta_b*f*(h_pr/cp)) / (1 + f);

    % Total pressure loss
    p04 = pi_b*p02;

    % Nozzle (4 -> 6), assume ideally expanded: p6 ≈ pa
    p6 = pa;

    % Check feasibility: must have p0,4 > p6 to expand
    if p04 <= p6
        error("Infeasible nozzle expansion: p0,4 (%.1f Pa) <= p6 (%.1f Pa). Check pi_d/pi_b/M.", p04, p6);
    end

    % Isentropic exit static temperature
    T6s = T04*(p6/p04)^((gamma-1)/gamma);

    % Nozzle efficiency: V6 = sqrt(2*eta_n*cp*(T04 - T6s))
    dT = T04 - T6s;
    if dT <= 0
        error("Non-positive nozzle enthalpy drop: T04 - T6s <= 0 (%.4f). Check inputs.", dT);
    end
    V6 = sqrt(2*eta_n*cp*dT);

    % Package outputs
    out = struct();
    out.M     = M;
    out.Ta    = Ta;
    out.pa    = pa;
    out.gamma = gamma;
    out.cp    = cp;
    out.R     = R;

    out.a     = a;
    out.Va    = Va;
    out.rho_a = rho_a;

    out.T0a   = T0a;
    out.p0a   = p0a;

    out.pi_d  = pi_d;
    out.pi_b  = pi_b;
    out.eta_b = eta_b;
    out.eta_n = eta_n;

    out.T02   = T02;
    out.p02   = p02;

    out.f     = f;
    out.h_pr  = h_pr;

    out.T04   = T04;
    out.p04   = p04;

    out.p6    = p6;
    out.T6s   = T6s;
    out.V6    = V6;

    % Thrust-per-(mdot_a) factor: (1+f)V6 - Va
    out.thrust_per_mdot = (1 + f)*V6 - Va;
end

function perf = ramjet_with_inlet(out, d_inlet)
% Computes mdot_a and thrust given inlet capture diameter d_inlet.

    Acap   = pi*(d_inlet^2)/4;
    mdot_a = out.rho_a*out.Va*Acap;

    % Momentum thrust (neglect pressure thrust)
    T = mdot_a * out.thrust_per_mdot;

    mdot_f = out.f * mdot_a;

    perf = struct();
    perf.d_inlet = d_inlet;
    perf.Acap    = Acap;
    perf.mdot_a  = mdot_a;
    perf.mdot_f  = mdot_f;
    perf.T       = T;
end

function d_req = ramjet_size_inlet_for_thrust(out, T_target)
% Solves for inlet diameter d required to achieve a target thrust,
% using mdot_a = rho*Va*A, thrust = mdot_a*[(1+f)V6 - Va].

    K = out.thrust_per_mdot; % [m/s]
    if K <= 0
        error("No positive thrust possible: (1+f)V6 - Va = %.3f <= 0. Increase f/T04 or reduce Mach/losses.", K);
    end

    mdot_req = T_target / K;  % [kg/s]
    Acap_req = mdot_req / (out.rho_a*out.Va); % [m^2]
    if Acap_req <= 0
        error("Computed negative/zero capture area. Check inputs.");
    end

    d_req = sqrt(4*Acap_req/pi);
end
