clc
clear
close all

n = 150; % Enter number of desired points

DOE = lhsdesign(n,6,'Criterion','correlation');

% c1 = 40*DOE(:,1)+420; % mainplane chord [mm]
% c2 = 40*DOE(:,2)+200; % 2nd element chord [mm]
% c3 = 30*DOE(:,3)+150; % 3rd element chord [mm]
% aoa1 = 15*DOE(:,4)-3; % mainplane AOA [deg]
% aoa2 = 13*DOE(:,5)+30; % 2nd element AOA [deg]
% aoa3 = 13*DOE(:,6)+45; % 3rd element AOA [deg]
% gap1 = 8*DOE(:,7)+17; % mp to 2nd gap [mm]
% gap2 = 8*DOE(:,8)+17; % 2nd to 3rd gap [mm]
% gurn = 10*DOE(:,9)+15; % gurney height [mm]

c1 = 425*ones(n,1); % mainplane chord [mm]
c2 = 220*ones(n,1); % 2nd element chord [mm]
c3 = 160*ones(n,1); % 3rd element chord [mm]
aoa1 = 3*DOE(:,1)+8; % mainplane AOA [deg]
aoa2 = 15*DOE(:,2)+20; % 2nd element AOA [deg]
aoa3 = 30*DOE(:,3)+30; % 3rd element AOA [deg]
gap1 = 15*DOE(:,4)+5; % mp to 2nd gap [mm]
gap2 = 15*DOE(:,5)+5; % 2nd to 3rd gap [mm]
gurn = 15*DOE(:,6)+5; % gurney height [mm]

% rle_mp = 0.08*DOE(:,1)+.005; % leading edge radius
% xp_mp = 0.2*DOE(:,2)+0.4; % pressure crest x
% yp_mp = -0.12*DOE(:,3); % pressure crest y
% cp_mp = -1.5*DOE(:,4); % pressure curvature
% xs_mp = 0.3*DOE(:,5)+0.3; % suction crest x
% ys_mp = (yp_mp+0.18).*DOE(:,6)-0.24; % suction crest y
% cs_mp = 2*DOE(:,7)-2.5; % suction curvature

% rle_fl = 0.08*DOE(:,8)+.005; % leading edge radius
% xp_fl = 0.2*DOE(:,9)+0.4; % pressure crest x
% yp_fl = -0.12*DOE(:,10); % pressure crest y
% cp_fl = -1.5*DOE(:,11); % pressure curvature
% xs_fl = 0.3*DOE(:,12)+0.3; % suction crest x
% ys_fl = (yp_mp+0.18).*DOE(:,13)-0.24; % suction crest y
% cs_fl = 2*DOE(:,14)-2.5; % suction curvature

% rle_mp = round(rle_mp,3);
% xp_mp = round(xp_mp,2);
% yp_mp = round(yp_mp,2);
% cp_mp = round(cp_mp,1);
% xs_mp = round(xs_mp,2);
% ys_mp = round(ys_mp,2);
% cs_mp = round(cs_mp,1);

% rle_fl = round(rle_fl,3);
% xp_fl = round(xp_fl,2);
% yp_fl = round(yp_fl,2);
% cp_fl = round(cp_fl,1);
% xs_fl = round(xs_fl,2);
% ys_fl = round(ys_fl,2);
% cs_fl = round(cs_fl,1);

% rle_mp = [0.03; rle_mp];
% xp_mp = [0.48; xp_mp];
% yp_mp = [-0.062; yp_mp];
% cp_mp = [-0.75; cp_mp];
% xs_mp = [0.37; xs_mp];
% ys_mp = [-0.18; ys_mp];
% cs_mp = [-1.4; cs_mp];

% rle_fl = [0.03; rle_fl];
% xp_fl = [0.48; xp_fl];
% yp_fl = [-0.062; yp_fl];
% cp_fl = [-0.75; cp_fl];
% xs_fl = [0.37; xs_fl];
% ys_fl = [-0.18; ys_fl];
% cs_fl = [-1.4; cs_fl];

c1 = [425; c1];
c2 = [220; c2];
c3 = [160; c3];
aoa1 = [9; aoa1];
aoa2 = [25; aoa2];
aoa3 = [40; aoa3];
gap1 = [14; gap1];
gap2 = [18; gap2];
gurn = [10; gurn];

run = (0:n)';

c1 = round(c1,0);
c2 = round(c2,0);
c3 = round(c3,0);
aoa1 = round(aoa1,1);
aoa2 = round(aoa2,1);
aoa3 = round(aoa3,1);
gap1 = round(gap1,1);
gap2 = round(gap2,1);
gurn = round(gurn,1);

writematrix([run c1 c2 c3 aoa1 aoa2 aoa3 gap1 gap2 gurn],'Run_List_v4.csv')
% Each row of DOE corresponds to a design point
% Each column of DOE corresponds to a variable