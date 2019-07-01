function L=GLM_LL(Spikes,X,k,b,fname)
    % INPUTS
    % Spikes:   row vector of length N with numbers of spikes at each time step
    % X:        raw (unfiltered) input. Matrix with dimensions DxN where D is
    %           the dimension of the input
    % k:        input filter coefficients (dimensionless)
    % b:        background firing rate Hz
    % fname:    a string indicating the nonlinearity to use 'quad', 'smooth', or 'exp'
    %
    % OUTPUTS
    % L:        loglikelihood of the model

    %% Some initialization
    % Choice of nonlinearity and its inverse
    switch fname
        case 'exp'
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % the exponential and its inverse
            'your code goes here';
            f = @(x) exp(x);
            invf= @(x) log(x);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'smooth'
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % the smooth rectified linear and its inverse
            'your code goes here';
            f= @(x) log(1 + exp(x));
            invf= @(x) log(exp(x)-1);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'quad'
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % the rectified quadratic and its inverse
            'your code goes here';
            f= @(x) (x.*(x>0)).^2+1e-5;
            invf= @(x) 1e-5-sqrt(x.*(x>0));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    % Time step size
    dt=.001; % seconds
    
    % Nonzero spike times (you'll want this!)
    nz=Spikes>0;
    
    % These are part of the constant term for a Poisson likelihood, we
    % don't need them for maximization, but we can use them to get the
    % exact loglikelihoods
    logfactorials=zeros(1,max(Spikes)+1);
    for i=1:length(logfactorials)
        logfactorials(i)=sum(log(1:i-1));
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculate log-likelihood and include the constant terms: log(p(y))=y*log(f(k*x+invf(b))*dt)-f(k*x+invf(b))*dt-log(y!)
    'your code goes here';
    L= sum(Spikes(nz).*log(f(k.'*X(:,nz))*dt) - f(k.'*X(:,nz))*dt - logfactorials(Spikes(nz)));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end