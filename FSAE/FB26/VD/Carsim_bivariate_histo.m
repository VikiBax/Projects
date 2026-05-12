data = readtable("2026-03-11_17.17.17.csv");

histogram(data.Beta)
%histogram2(data.Roll, data.Pitch)
xlabel("Roll")
ylabel("Pitch")