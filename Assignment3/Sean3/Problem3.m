load('Problem3.mat')
[tau,E,k,Vsig]=fitLIF(V,Spikes,X);

% Running the funciton gives: 
% tau = 0.0097, 
% E = 8.7430
% Vsig = 0.0144

% plotting k
plot(k);
title('k (below), tau = 0.0097, E = 8.7430, Vsig = 0.0144');