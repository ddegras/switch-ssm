function [Aboot,Qboot,muboot,Sigmaboot,Piboot,Zboot,LLboot] = ... 
    bootstrap_var(A,Q,mu,Sigma,Pi,Z,T,B,control,equal,fixed,scale,parallel)

%-------------------------------------------------------------------------%
%             BOOTSTRAP RESAMPLING FOR SWITCHING VAR MODEL                %
%-------------------------------------------------------------------------%


% Check number of arguments
narginchk(7,13);

% Initialize missing arguments if needed
if ~exist('B','var')
    B = 100;
end
if ~exist('control','var')
    control = [];
end
if ~exist('equal','var')
    equal = [];
end
if ~exist('fixed','var')
    fixed = [];
end
if ~exist('scale','var')
    scale = [];
end
if ~exist('parallel','var')
    parallel = false;
end

% Model dimensions
[r,~,p,M] = size(A);

% Bootstrap estimates
Aboot = zeros(r,r,p,M,B);
Qboot = zeros(r,r,M,B);
muboot = zeros(r,M,B);
Sigmaboot = zeros(r,r,M,B);
Piboot = zeros(M,B);
Zboot = zeros(M,M,B);
LLboot = zeros(B,1);
warning('off');



if parallel     % PARALLEL LOOP

    parfor b=1:B 
        parfor_progress(B);
        % Parametric bootstrap 
        m = mu; 
        Sig = Sigma;
        Q_ = Q;
        Z_ = Z;
        x = zeros(r,T);
        St = 0;
        Amat = reshape(A,[r,p*r,M]);
        for t=1:T       
            % Simulate regime S(t)
            if t == 1
                c = cumsum(Pi);
            else        
                Stm1 = St;
                c = cumsum(Z_(Stm1,:));
            end
            rbt = rand(1);              
            St = M+1-sum(rbt <= c);        
            % Simulate observation vector x(t)
            if t <= p
                x(:,t) = mvnrnd(m(:,St)',Sig(:,:,St))';
            else
                Xtm1 = reshape(x(:,t-1:-1:t-p),p*r,1);
                vt = mvnrnd(zeros(1,r),Q_(:,:,St))';
                x(:,t) = Amat(:,:,St) * Xtm1 + vt;
            end        
        end       
        % EM  
        [~,~,~,~,Ab,Qb,mub,Sigmab,Pib,Zb,LL] = ... 
                switch_var(x,M,p,A,Q,mu,Sigma,Pi,Z,control,equal,fixed,scale);   
        Aboot(:,:,:,:,b) = Ab;
        Qboot(:,:,:,b) = Qb;
        muboot(:,:,b) = mub;
        Sigmaboot(:,:,:,b) = Sigmab;
        Piboot(:,b) = Pib;
        Zboot(:,:,b) = Zb;
        LLboot(b) = max(LL);
        parfor_progress;    
    end
    parfor_progress(0);
    
else        % SEQUENTIAL LOOP

    for b=1:B  
       % Parametric bootstrap 
        x = zeros(r,T);
        Amat = reshape(A,[r,p*r,M]);
        for t=1:T       
            % Simulate regime S(t)
            if t == 1
                c = cumsum(Pi);
            else        
                Stm1 = St;
                c = cumsum(Z(Stm1,:));
            end
            rbt = rand(1);              
            St = M+1-sum(rbt <= c);        
            % Simulate observation vector x(t)
            if t <= p
                x(:,t) = mvnrnd(mu(:,St)',Sigma(:,:,St))';
            else
                Xtm1 = reshape(x(:,t-1:-1:t-p),p*r,1);
                vt = mvnrnd(zeros(1,r),Q(:,:,St))';
                x(:,t) = Amat(:,:,St) * Xtm1 + vt;
            end        
        end       
        % EM 
        [~,~,~,~,Ab,Qb,mub,Sigmab,Pib,Zb,LL] = ... 
                switch_var(x,M,p,A,Q,mu,Sigma,Pi,Z,control,equal,fixed,scale);   
        Aboot(:,:,:,:,b) = Ab;
        Qboot(:,:,:,b) = Qb;
        muboot(:,:,b) = mub;
        Sigmaboot(:,:,:,b) = Sigmab;
        Piboot(:,b) = Pib;
        Zboot(:,:,b) = Zb;
        LLboot(b) = max(LL);   
    end
end


    