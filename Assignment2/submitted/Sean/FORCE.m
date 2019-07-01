function [output,network]=FORCE(prelearn_time,learn_time,test_time,N,use_hints,seed,make_plot)
    %% Parameters
    % Network parameters
    network.f=@tanh;  % tanh nonlinearity
    network.g=1.5;  % spectral radius of the initial weight matrix
    network.p=0.1;  % probability of connection in initial weight matrix
    network.dt=0.001;  % time-step in seconds
    network.tau=0.01;  % time constant in seconds
    network.sigma=0.001;  % noise level
    network.alpha=1;  % radius of initial covariance matrix
    network.numR=3;  % number of recurrent units to print
    plot_time = 0.25;  % time between plot updates
    plot_window = 5;  % size of plot window
    
    % Target function and hint parameters
    period = 1;  % time in seconds
    pos1 = 0.41;  % position of camel hump relative to period
    pos2 = 0.59;  % position of camel hump relative to period
    width = 0.06;  % camel hump widths
    %% Initialization
    % Network parameters
    network.N=N;  % number of neurons

    % Set the seed
    rng(seed);
    
    % Simulation times
    T_prelearn=round(prelearn_time/network.dt);  % number of time-steps before learning
    T_learn=round(learn_time/network.dt);  % number of time-steps during learning
    T_test=round(test_time/network.dt);  % number of time-steps after learning
    T=T_prelearn + T_learn + T_test;  % total number of time-steps
    
    % Some constants
    dt_tau=network.dt/network.tau;  % useful for Euler steps
    sigma=network.sigma*sqrt(dt_tau*(2-dt_tau));  % convertion for the noise level to match the Euler integration (see description in problem 1)

    % Initialize the output function
    T_period = round(period/network.dt);  % number of time-steps per target function period
    fO = @(t)exp(-.5*(mod(t,T_period)-pos1*T_period).^2/(T_period*width)^2)+exp(-.5*(mod(t,T_period)-pos2*T_period).^2/(T_period*width)^2);
    O=1;
    if use_hints
        H=2;
    else
        H=1;
    end

    % Initialize the network weights    
    network.w0=randn(network.N)*network.g/sqrt(network.p*network.N);  % random matrix where each weight has a mean of network.g/sqrt(network.p*network.N)
    network.w0(rand(network.N^2,1)>network.p)=0;  % set all but a fraction p of the weights to zeros
    network.wF=randn(network.N,O+H)/sqrt(O);  % feedback weights

    % Initialize for learning
    network.wO=zeros(O+H,network.N);  % output weights (need to have hint dimensions as well)
    network.P=eye(network.N)/network.alpha;  % initial inverse covariance matrix
    
    % Initialize network activity
    network.x=randn(network.N,1);  % initial value of x

    % Initialize the output
    output.fO=zeros(O,T);  % target function
    output.z=zeros(O,T);  % what the network actually generates
    output.r=zeros(network.numR,T);  % a few of the recurrent neurons
    output.h=zeros(H,T);  % hint function if any
    
    % Initialize plotting
    if make_plot
        T_plot = round(plot_time/network.dt);  % number of time-steps between plot updates
        T_window = round(plot_window/network.dt);  % number of time-step in plot window
        f=figure; 
        clear a;
        a(1) = subplot(3,1,1,'Parent',f);
        a(2) = subplot(3,1,2,'Parent',f);
        a(3) = subplot(3,1,3,'Parent',f);
    end
    
    %% Main loop
    for t=1:T
        
        % Plotting
        if make_plot && mod((t-1),T_plot)==0 && t>1
            min_t=max(1,t-T_window);
            cla(a(1));
            plot(a(1),network.dt*(min_t:t-1),output.r(:,min_t:t-1)')
            set(a(1),'XLim',[min_t-1 t-1]*network.dt,'YLim',[-1.1 1.1])
            cla(a(2));
            hold(a(2),'on');
            plot(a(2),network.dt*(min_t:t-1),output.fO(:,min_t:t-1)','k')
            plot(a(2),network.dt*(min_t:t-1),output.z(:,min_t:t-1)')
            set(a(2),'XLim',[min_t-1 t-1]*network.dt,'YLim',[-0.05 1.05])
            cla(a(3));
            plot(a(3),network.dt*(min_t:t-1),output.h(:,min_t:t-1)')
            set(a(3),'XLim',[min_t-1 t-1]*network.dt,'YLim',[-1.1 1.1])
            linkaxes(a,'x');
        end

        % Get target and hint (if using hints)
        if use_hints
            angle = t*2*pi/T_period;
            hint = [cos(angle);sin(angle)];
        else
            hint = 0;
        end
        fO_h=[fO(t);hint]; % this has both the target and the hints (if any)
        
        % Apply nonlinearity
        r=network.f(network.x); % rates
      
        % Learning step
        if T_prelearn < t && t <= T_learn+T_prelearn
            network.P= network.P - network.P*r*r.'* network.P/(1+r.'*network.P*r);
            network.wO= network.wO - (network.wO*r - fO_h)*r.'*network.P;
 % This is where you should insert the lines of code that update the output weights network.wO.
 % To do so, you will first have to update network.P, the inverse covariance matrix.
 % See Equations 5 and 6 in problem 1.

        end

        % Run the dynamics
        network.x = network.x*(1 - dt_tau) + dt_tau*(network.w0*r+ network.wF*fO_h);
 % This is where you insert the line of code for an Euler step. See Equation 2 in problem 1.
 % You can ignore adding the noise; I do it for you on the next line

        network.x=network.x + sigma*randn(network.N,1);  % add on a bit of noise; note that I already multiplied sigma by the correct prefactor above

        % Save the outputs
        output.fO(:,t)=fO_h(1:O); % save target
        output.z(:,t)=network.wO(1:O,:)*r; % save output
        output.r(:,t)=r(1:network.numR); % save resevoir units
        output.h(:,t)=fO_h(O+1:end); % save hint
    end
    
    %% Calculate error
    % Since periodic output slowly phase shift, you need to scan across a
    % whole period to find the best alignment and then report that value as
    % the error
    z = output.z(:,T-T_period+1:T);
    min_nrmse = inf;
    for offset=0:T_period-1
        tar = output.fO(:,T-T_period+1-offset:T-offset);
        err = z-tar;
        nrmse = sqrt(mean(sum(err.^2,1)) / mean(sum(tar.^2,1)));
        if nrmse < min_nrmse
            min_nrmse = nrmse;
        end
    end
    output.nrmse=min_nrmse;
end