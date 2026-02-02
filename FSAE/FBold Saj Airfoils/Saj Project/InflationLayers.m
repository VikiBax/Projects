clc
clear
close all

%% Inputs
yplus = .5; % target y+ (should be between 1-5 or 30+)
G = 1.2; % growth rate (1.2 is fine)

% make sure these parameters match the simulation parameters and geometry
rho = 1.225; % ambient density (kg/m^3)
V = 15; % airspeed (m/s)
L = 1; % maximum chord length (m)
mu = 1.789e-5; % ambient viscosity (kg/m*s)

%% Outputs
ReL = rho*V*L/mu;
CfL = .0592/(ReL^0.2);
tau = .5*rho*V^2*CfL;
delta = 0.37*L/(ReL^0.2)

uT = sqrt(tau/rho);
yp = yplus*mu/(rho*uT);

yH = 2*yp % first layer height
AR = .005/yH % first layer aspect ratio
N = ceil(log(1-(delta*(1-G)/yH))/log(G)) % number of inflation layers

n = 6;

SR = 1/sum(G.^(0:n-1))*sum(G.^(0:n-2));