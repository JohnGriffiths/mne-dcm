function [Y] = slim_DEMO_model_reduction_ERP
% Test script for a slimmed down minimal dcm erp
%
% Based on DEMO_model_reduction_ERP.m
%
% Uses slimmed down modifications of certain key dcm functions
%
% The slimmed down functions are defined at the bottom 
% of this function, and are names the same as the original 
% functions but with the prefix 'slim_' rather than 'spm_'
%
% The removed bits in the slimmed down functions are mostly
% 'if' and 'catch' statements that check for or use conditions 
% or variables that are not actually present in this demo
%
%
%
% JG,DD June 2017
%
%__________________________________________________________________________



% model specification – a simple two source model with two electrodes
% =========================================================================
rng('default')

Nc    = 2;                                        % number of channels
Ns    = 2;                                        % number of sources

options.spatial  = 'LFP';
options.model    = 'ERP';
options.analysis = 'ERP';
M.dipfit.model   = options.model;
M.dipfit.type    = options.spatial;
M.dipfit.Nc      = Nc;
M.dipfit.Ns      = Ns;

% specify connectivity – reciprocal connections with condition specific
% changes in intrinsic and extrinsic connectivity
%--------------------------------------------------------------------------
A{1}    = [0 0; 1 0];
A{2}    = [0 1; 0 0];
A{3}    = [0 0; 0 0];
B{1}    = [1 0; 1 0];
C       = [1; 0];

% HARD CODE??
[pE,pC] = slim_dcm_neural_priors(A,B,C,options.model);
[gE,gC] = slim_L_priors(M.dipfit);
[x,f]   = slim_dcm_x_neural(pE,options.model);   %% this is coded

% hyperpriors (assuming a high signal to noise)
%--------------------------------------------------------------------------
hE      = 6;
hC      = 1/128;

% create model
%--------------------------------------------------------------------------
M.IS   = 'slim_gen_erp' ;
M.G    = 'slim_lx_erp';
M.f    = f;
M.x    = x;
M.pE   = pE;
M.pC   = pC;
M.gE   = gE;
M.gC   = gC;
M.hE   = hE;
M.hC   = hC;
M.m    = length(B);
M.n    = length(spm_vec(M.x));
M.l    = Nc;
M.ns   = 64;

% create input structure
%--------------------------------------------------------------------------
dt     = 4/1000;
pst    = (1:M.ns)*dt;
M.ons  = 64;
M.dur  = 16;
U.dt   = dt;
U.X    = [0; 1];

% specified true connectivity (P) and spatial parameters (G) – with
% condition specific effects on the intrinsic connectivity of the first
% source and its forward extrinsic connection
%--------------------------------------------------------------------------
P      = pE;
G      = gE;
P.B{1} = [-1/4 0; 1/2 0];


% generate neuronal response and data
%--------------------------------------------------------------------------
x     = slim_gen_erp(P,M,U);                 % neuronal response
L     = slim_lx_erp(G,M.dipfit);             % lead field
V     = slim_sqrtm(slim_Q(1/2,M.ns));         % square root of noise covariance
for i = 1:length(x)
    n    = exp(-hE/2)*V*randn(M.ns,Nc);     % noise
    s{i} = x{i}*L';                         % signal
    y{i} = s{i} + n;                        % data (signal plus noise)
end

% data structure specification
%--------------------------------------------------------------------------
Y.y   = y;
Y.Q   = {slim_Q(1/2,M.ns,1)};
Y.dt  = dt;
Y.pst = pst;



%--------------------------------------------------------------------------


% SLIMMED FUNCTIONS:
% =================



function [y,pst] = slim_gen_erp(P,M,U)
    % Generates a prediction of trial-specific source activity
    
    % check input u = f(t,P,M) and switch off full delay operator
    %--------------------------------------------------------------------------
    try, M.fu; catch, M.fu  = 'spm_erp_u'; end
    try, M.ns; catch, M.ns  = 128;         end
    try, M.N;  catch, M.N   = 0;           end
    try, U.dt; catch, U.dt  = 0.004;       end
    
    % peristimulus time
    %--------------------------------------------------------------------------
    % if nargout > 1
    %     pst = (1:M.ns)*U.dt - M.ons/1000;
    % end
    
    % within-trial (exogenous) inputs
    %==========================================================================
    %if ~isfield(U,'u')
    
    % peri-stimulus time inputs
    %----------------------------------------------------------------------
    U.u = feval(M.fu,(1:M.ns)*U.dt,P,M);
    
    % end
    
    % if isfield(M,'u')
    
    % remove M.u to preclude endogenous input
    %----------------------------------------------------------------------
    %     M = rmfield(M,'u');
    
    % end
    
    % between-trial (experimental) inputs
    %==========================================================================
    %if isfield(U,'X')
    X = U.X;
    % else
    %     X = sparse(1,0);
    % end
    %
    % if ~size(X,1)
    %     X = sparse(1,0);
    % end
    
    
    % cycle over trials
    %==========================================================================
    y      = cell(size(X,1),1);
    for  c = 1:size(X,1)
        
        % condition-specific parameters
        %----------------------------------------------------------------------
        Q   = slim_gen_Q(P,X(c,:));
        
        % solve for steady-state - for each condition
        %----------------------------------------------------------------------
        M.x  = slim_dcm_neural_x(Q,M);
        
        % integrate DCM - for this condition
        %----------------------------------------------------------------------
        %y{c} = slim_int_L(Q,M,U);
        y{c} = slim_int_L(Q,M,U);
        
    end
    
    
function [Q] = slim_gen_Q(P,X)
    %% this is really just constructing Q field by field - I deleted the lines that are not evaluated for this demo
    Q = rmfield(P,'B');
    
    % trial-specific effects on A (connections)
    for i = 1:length(X)
        
        % extrinsic (driving) connections
        %----------------------------------------------------------------------
        for j = 1:length(Q.A)
            Q.A{j} = Q.A{j} + X(i)*P.B{i};
        end
        
        Q.G(:,1) = Q.G(:,1) + X(i)*diag(P.B{i});
        
    end


function [x] = slim_dcm_neural_x(P,M)
%% Returns the fixed point or steady-state of a neural mass DCM (only for cnductance based models)
x    = M.x;
 
function [y] = slim_int_L(P,M,U)
    % Integrate a MIMO nonlinear system using a fixed Jacobian: J(x(0))
    % FORMAT [y] = spm_int_L(P,M,U)
    % P   - model parameters
    % M   - model structure
    % U   - input structure or matrix
    %
    % y   - (v x l)  response y = g(x,u,P)
    
    dt = U.dt;
    
    % Initial states and inputs
    x   = M.x;
    u   = U.u(1,:);
    
    % add [0] states if not specified
    f   = str2func(M.f);
    M.f = f;
    
    % output nonlinearity, if specified
    
    g = @(x,u,P,M) x;
    M.g = g;
    
    % dx(t)/dt and Jacobian df/dx and check for delay operator
    %--------------------------------------------------------------------------
    D       = 1;
    n       = numel(x);
    [fx,dfdx,D] = f(x,u,P,M);
    
    OPT.tol = 1e-6*norm((dfdx),'inf');
    
    p = abs(eigs(dfdx,1,'SR',OPT));
    
    N     = ceil(max(1,dt*p*2));
    Q     = (spm_expm(dt*D*dfdx/N) - speye(n,n))*spm_inv(dfdx);
    
    % integrate
    v     = spm_vec(x);
    for i = 1:size(U.u,1)
        
        % input
        u  = U.u(i,:);
        
        for j = 1:N
            v = v + Q*f(v,u,P,M);
        end
        
        % output - implement g(x)
        y(:,i) = g(v,u,P,M);
        
    end
    
    % transpose
    %--------------------------------------------------------------------------
    y      = real(y');
    

function [x,f,h] = slim_dcm_x_neural(P,model)
    % Returns the state and equation of neural mass models - case{'erp'}
    % linear David et al model (linear in states)
    disp(model)
    n  = length(P.A{1});                          % number of sources
    m  = 9;                                       % number of states
    x  = sparse(n,m);
    f  = 'slim_fx_erp';
    
    
function [pE,pC] = slim_dcm_neural_priors(A,B,C,model)
    % Prepares the priors on the parameters of neural mass models
    %     case{'erp','sep'}
    [pE,pC] = slim_erp_priors(A,B,C);
    

function [E,V] = slim_erp_priors(A,B,C)
    % prior moments for a neural-mass model of ERPs
    
    % default: a single source model
    N     = 4;
    n     = size(C,1);                                % number of sources
    u     = size(C,2);                                % number of inputs
    
    % parameters for neural-mass forward model
    %==========================================================================
    
    % set intrinsic [excitatory] time constants and gain
    %--------------------------------------------------------------------------
    E.T   = sparse(n,2);  V.T = sparse(n,2) + 1/16;   % time constants
    E.G   = sparse(n,2);  V.G = sparse(n,2) + 1/16;   % synaptic density
    
    % set parameter of activation function
    %--------------------------------------------------------------------------
    E.S   = [0 0];        V.S = [1 1]/16;             % dispersion & threshold
    
    
    % set extrinsic connectivity
    %--------------------------------------------------------------------------
    Q     = sparse(n,n);
    for i = 1:length(A)
        A{i} = ~~A{i};
        E.A{i} = A{i}*N - N;                          % forward
        V.A{i} = A{i}/16;                             % backward
        Q      = Q | A{i};                            % and lateral connections
    end
    
    for i = 1:length(B)
        B{i} = ~~B{i};
        E.B{i} = 0*B{i};                              % input-dependent scaling
        V.B{i} = B{i}/8;
        Q      = Q | B{i};
    end
    C      = ~~C;
    E.C    = C*N - N;                                 % where inputs enter
    V.C    = C/32;
    
    % set intrinsic connectivity
    %--------------------------------------------------------------------------
    E.H    = sparse(1,4);
    V.H    = sparse(1,4) + 1/16;
    
    % set (extrinsic) delay
    %--------------------------------------------------------------------------
    E.D    = sparse(n,n);
    V.D    = Q/16;
    
    % fix intrinsic delays
    %--------------------------------------------------------------------------
    V.D    = V.D - diag(diag(V.D));
    
    % set stimulus parameters: onset, dispersion and sustained proportion
    %--------------------------------------------------------------------------
    E.R    = sparse(u,2);  V.R   = E.R + 1/16;
    return
    
function [pE,pC] = slim_L_priors(dipfit)
    % prior moments for the lead-field parameters of ERP models
    % defaults
    %--------------------------------------------------------------------------
    model    = dipfit.model;
    type     = dipfit.type;     %ok
    location = 0;
    pC       = [];
    
    % number of sources
    n = dipfit.Ns;
    m = dipfit.Nc;
    
    % location priors (4 mm)
    V = 0;
    
    % parameters for electromagnetic forward model
    %     case{'LFP'}
    pE.Lpos = sparse(3,0);   pC.Lpos = sparse(3,0);    % positions
    pE.L    = ones(1,m);     pC.L    = ones(1,m)*64;   % gains
    
    
    % contributing states (encoded in J)
    if ischar(model), mod.source = model; model = mod; end
    pE.J = {};
    pC.J = {};
    
    %         case{'ERP','SEP'}
    %--------------------------------------------------------------
    pE.J{end + 1} = sparse(1,9,1,1,9);               % 9 states
    pC.J{end + 1} = sparse(1,[1 7],1/32,1,9);
    pE.J = pE.J{:};
    pC.J = pC.J{:};

function [Q] = slim_Q(varargin)   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    a = varargin{1}
    n = varargin{2}
    if length(varargin) == 2
        [Q] = spm_Q(a,n);
    elseif length(varargin) == 3
        q = varargin{3}
        [Q] = spm_Q(a,n,q)
    end
    

function [L] = slim_lx_erp(P,dipfit)
    % observer matrix for a neural mass model: y = G*x
    % parameterised lead field times source contribution to ECD
    L = spm_erp_L(P,dipfit);               % lead field per source
    L = kron(P.J,L);                       % lead-field per state


function [L] = spm_erp_L(P,dipfit)
    % returns [projected] lead field L as a function of position and moments
    % type of spatial model and modality
    type = dipfit.type;
    %     case{'LFP'}
    m     = length(P.L);
    n = dipfit.Ns;
    L     = sparse(1:m,1:m,P.L,m,n);
    
function [K] = slim_sqrtm(V)
    [u,s] = spm_svd(V,0);
    s     = sqrt(abs(diag(s)));
    m     = length(s);
    s     = sparse(1:m,1:m,s);
    K     = u*s*u';
        
        
  
function [f,J,Q] = slim_fx_erp(x,u,P,M)
% state equations for a neural mass model of erps

% get dimensions and configure state variables
%--------------------------------------------------------------------------
x  = spm_unvec(x,M.x);      % neuronal states
n  = size(x,1);             % number of sources

% [default] fixed parameters
%--------------------------------------------------------------------------
E = [1 1/2 1/8]*32;         % extrinsic rates (forward, backward, lateral)
G = [1 4/5 1/4 1/4]*128;    % intrinsic rates (g1 g2 g3 g4)
D = [2 16];                 % propogation delays (intrinsic, extrinsic)
H = [4 32];                 % receptor densities (excitatory, inhibitory)
T = [8 16];                 % synaptic constants (excitatory, inhibitory)
R = [2 1]/3;                % parameters of static nonlinearity


% test for free parameters on intrinsic connections
%--------------------------------------------------------------------------
try
    G = G.*exp(P.H);
end
G     = ones(n,1)*G;

% exponential transform to ensure positivity constraints
%--------------------------------------------------------------------------
A{1} = exp(P.A{1})*E(1);
A{2} = exp(P.A{2})*E(2);
A{3} = exp(P.A{3})*E(3);
C     = exp(P.C);

% intrinsic connectivity and parameters
%--------------------------------------------------------------------------
Te    = T(1)/1000*exp(P.T(:,1));         % excitatory time constants
Ti    = T(2)/1000*exp(P.T(:,2));         % inhibitory time constants
He    = H(1)*exp(P.G(:,1));              % excitatory receptor density
Hi    = H(2)*exp(P.G(:,2));              % inhibitory receptor density

% pre-synaptic inputs: s(V)
%--------------------------------------------------------------------------
R     = R.*exp(P.S);
S     = 1./(1 + exp(-R(1)*(x - R(2)))) - 1./(1 + exp(R(1)*R(2)));

% input

% exogenous input
%----------------------------------------------------------------------
U = C*u(:)*2;


% State: f(x)
%==========================================================================

% Supragranular layer (inhibitory interneurons): Voltage & depolarizing current
%--------------------------------------------------------------------------
f(:,7) = x(:,8);
f(:,8) = (He.*((A{2} + A{3})*S(:,9) + G(:,3).*S(:,9)) - 2*x(:,8) - x(:,7)./Te)./Te;

% Granular layer (spiny stellate cells): Voltage & depolarizing current
%--------------------------------------------------------------------------
f(:,1) = x(:,4);
f(:,4) = (He.*((A{1} + A{3})*S(:,9) + G(:,1).*S(:,9) + U) - 2*x(:,4) - x(:,1)./Te)./Te;

% Infra-granular layer (pyramidal cells): depolarizing current
%--------------------------------------------------------------------------
f(:,2) = x(:,5);
f(:,5) = (He.*((A{2} + A{3})*S(:,9) + G(:,2).*S(:,1)) - 2*x(:,5) - x(:,2)./Te)./Te;

% Infra-granular layer (pyramidal cells): hyperpolarizing current
%--------------------------------------------------------------------------
f(:,3) = x(:,6);
f(:,6) = (Hi.*G(:,4).*S(:,7) - 2*x(:,6) - x(:,3)./Ti)./Ti;

% Infra-granular layer (pyramidal cells): Voltage
%--------------------------------------------------------------------------
f(:,9) = x(:,5) - x(:,6);
f      = spm_vec(f);

if nargout < 2;return, end

% Jacobian
%==========================================================================
J  = spm_diff(M.f,x,u,P,M,1);

% delays
De = D(2).*exp(P.D)/1000;
Di = D(1)/1000;
De = (1 - speye(n,n)).*De;
Di = (1 - speye(9,9)).*Di;
De = kron(ones(9,9),De);
Di = kron(Di,speye(n,n));
D  = Di + De;
%--------------------------------------------------------------------------
Q  = inv(speye(length(J)) + D.*J);
 
