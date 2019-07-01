function [tau,E,k,Vsig]=fitLIF(V,Spikes,X)
    % INPUTS
    % V:        membrane potential (V). row vector of length N, where N is the number
    %           of time-steps per trial
    % Spikes:   logical array, true for the time-steps when the spikes occur. Also
    %           row vector of length N
    % X:        raw (unfiltered) input (V; converted from amps via V=IR assuming
    %           total membrane resistance is 10e6 ohms). Matrix with dimensions DxN
    %           where D is the dimension of the input
    %
    % OUTPUTS
    % tau:      membrane time constant (s)
    % E:        reversal potential (V)
    % k:        input filter coefficients (dimensionless)
    % Vsig:     stochastic dynamics noise (V)
    
    %% Some initialization
    % simulation length
    N=length(V);
    
    % Time step size
    dt=.001; % seconds
    
    % Refractory period length
    Trefractory=.002; % seconds
    
    % Extract the start and end times of each inter-spike interval
    starts=find([false(1,round(Trefractory/dt)) Spikes(1:end-round(Trefractory/dt))]); % final time-step of each refractory period
    stops=find(Spikes); % time-step prior to each spike
    ISIs=[starts(1:end-1);stops(2:end)]; % array of ISIs discarding the unmatched first stop time and last start time
    
    % We won't use time steps when either V(t) or V(t-1) are given by
    % spiking mechanism. The following finds the good 'quiet' times
    quiet=false(1,N-1);
    for i=1:size(ISIs,2)  % find times between the refractory period and the next spike
        quiet(ISIs(1,i):ISIs(2,i)-2)=true;
    end
    
    %% Least squares parameter estimation
    % This quantity has a zero-mean Gaussian distribution:
    %   V(t)-[V(t-1);1;x(t)]'*params
    % where:
    %   params(1)=1-dt/tau
    %   params(2)=E*dt/tau
    %   params(3:end)=k*dt/tau;
    % So we can use least squares to get the parameters

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Construct the data matrix S
    S = vertcat(V,ones(1,N),X);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % These expressions will throw away the time steps during the spikes
    % and refractory periods that we don't want to use
    S_quiet=S(:,quiet); % throw away spike times from data matrix
    V_quiet=V(2:end); % Not using the first term of V
    V_quiet=V_quiet(quiet); % throw away spike times from voltage
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Solve for the params
    params = inv(S_quiet*S_quiet.')*S_quiet*V_quiet.';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Deal params into the return variables
    tau = dt/(1-params(1));
    E = params(1)*tau/dt;
    k = params(3:end)*tau/dt;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Estimate the stochastic dynamics noise term
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % determine Vsig here
    Vsig = sqrt(1/(dt*(N-1))*(V_quiet.'-S_quiet.'*params).'*(V_quiet.'-S_quiet.'*params));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end