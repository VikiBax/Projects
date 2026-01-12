function  Test = BoundaryCheck(Param)

%set tolerances and boundaries of Front Wing elements 2,3
xtol = 0; ytol = 0; 
xmax = 171.17 - xtol; ymax = 115.29 - ytol; 

%Known cord lengths of FW elements 2,3
w2c = 129.79324; w3c = 64.89662; 

%Xmax condition 
if w2c*cosd(Param(1)) + Param(2)*cosd(Param(3)) + w3c*cosd(Param(4)) + Param(5)*cosd(Param(6)) > xmax
    Test=false; %fail 
elseif w2c*sind(Param(1)) + Param(2)*sind(Param(3)) + w3c*sind(Param(4)) + Param(5)*sind(Param(6)) > ymax
    Test=false; %fail 
elseif Param(4) < Param(1)
    Test=false; %fail 
elseif abs(Param(4)-Param(1)) < 3
    Test=false; %fail



else 
    Test=true; %pass
end