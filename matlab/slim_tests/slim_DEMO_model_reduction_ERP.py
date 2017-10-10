
"""
NOTES:
======


- Python dictionaries will be used for matlab cell arrays

e.g. 

Matlab:   A{1} = ...
Python:   A[1] = ...


"""

import numpy as np



def main():
    
    """
    % model specification: a simple two source model with two electrodes
    % =========================================================================
    rng('default')
    """
    
    """
    Nc    = 2;                                        % number of channels
    Ns    = 2;                                        % number of sources
    """;
    Nc    = 2
    Ns    = 2

    """
    options.spatial  = 'LFP';
    options.model    = 'ERP';
    options.analysis = 'ERP';
    M.dipfit.model   = options.model;
    M.dipfit.type    = options.spatial;
    M.dipfit.Nc      = Nc;
    M.dipfit.Ns      = Ns;
    """
    options,M = {},{}
    options['spatial'] = 'LFP'
    options['model'] = 'ERP'
    options['analysis'] = 'ERP'
    M['dipfit'] = {'model': options['model'],
                   'type': options['spatial'],
                   'Nc': Nc,
                   'Ns': Ns}
    """
    % specify connectivity: reciprocal connections with condition specific
    % changes in intrinsic and extrinsic connectivity
    %--------------------------------------------------------------------------
    A{1}    = [0 0; 1 0];
    A{2}    = [0 1; 0 0];
    A{3}    = [0 0; 0 0];
    B{1}    = [1 0; 1 0];
    C       = [1; 0];
    """
    A,B,C = {},{},{}
    A[0] = np.array([[0,0],[1,0]])
    A[1] = np.array([[0,1],[0,0]])
    A[2] = np.array([[0,0],[0,0]])
    B[0] = np.array([[1,0],[1,0]])
    C[0] = np.array([[1,0]])

    """
    [pE,pC] = slim_dcm_neural_priors(A,B,C,options.model);
    [gE,gC] = slim_L_priors(M.dipfit);
    [x,f]   = slim_dcm_x_neural(pE,options.model);   %% this is coded
    """  

    pE,pC = slim_dcm_neural_priors(A,B,C,options['model'])
    gE,gC = slim_L_priors(M['dipfit'])
    x,f = slim_dcm_x_neural(pE,options['model'])


    """
    % hyperpriors (assuming a high signal to noise)
    %--------------------------------------------------------------------------
    hE      = 6;
    hC      = 1/128;
    """
    hE = 6
    hC = 1./128

    """
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
    """
    M['IS'] = 'slim_gen_erp'
    M['G'] = 'slim_lx_erp'
    M['f'] = 'f'
    M['x'] = 'x'
    M['pE'] = pE
    M['pC'] = pC
    M['gE'] = gE
    M['gC'] = gC
    M['hE'] = hE
    M['hC'] = hC
    M['m'] = len(B)
    M['n'] = len(spm_vec(M['x']))
    M['l'] = Nc
    M['ns'] = 64.


    """
    % create input structure
    %--------------------------------------------------------------------------
    dt     = 4/1000;
    pst    = (1:M.ns)*dt;
    M.ons  = 64;
    M.dur  = 16;
    U.dt   = dt;
    U.X    = [0; 1];
    """
    dt = 4/1000.
    pst = np.arange(1,M['ns']*dt)
    M['ons'] = 64
    M['dur'] = 16
    U['dt'] = dt
    U['X'] = np.array([0,1])


    """
    % specified true connectivity (P) and spatial parameters (G)  with
    % condition specific effects on the intrinsic connectivity of the first
    % source and its forward extrinsic connection
    %--------------------------------------------------------------------------
    P      = pE;
    G      = gE;
    P.B{1} = [-1/4 0; 1/2 0];
    """
    P = pE
    G = gE
    P['B'] = [np.array([[-1./4, 0],[1./2, 0]])]


    """
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
    """
     
    x = slim_gen_erp(P,M,U)
    L = slim_lx_erp(G,M['dipfit'])
    V = slim_sqrtm(slim_Q(1./2, M['ns']))
    for i in range(0,len(x)):
      n = np.exp(-hE/2.)*V*np.randn(M['ns'],Nc)
      s[i] = x[i]*L.T
      y[i] = s[i] + n   
    
    
    """
    % data structure specification
    %--------------------------------------------------------------------------
    Y.y   = y;
    Y.Q   = {slim_Q(1/2,M.ns,1)};
    Y.dt  = dt;
    Y.pst = pst;
    """
    Y['y'] = y
    Y['Q'] = [slim_Q(1./2,M['ns'],1)]
    Y['dt'] = dt
    Y['pst'] = pst




def slim_fx_erp(x,u,P,M):
    """
    % State equations for a neural mass model of erps
  
    Port of spm_fx_erp.m
  
    Usage: 
  
      f,J,D = spm_fx_erp(x,u,P,M,returnJ=True,returnD=True)
   
    x      - state vector
    x(:,1) - voltage (spiny stellate cells)
    x(:,2) - voltage (pyramidal cells) +ve
    x(:,3) - voltage (pyramidal cells) -ve
    x(:,4) - current (spiny stellate cells)    depolarizing
    x(:,5) - current (pyramidal cells)         depolarizing
    x(:,6) - current (pyramidal cells)         hyperpolarizing
    x(:,7) - voltage (inhibitory interneurons)
    x(:,8) - current (inhibitory interneurons) depolarizing
    x(:,9) - voltage (pyramidal cells)

    f        - dx(t)/dt  = f(x(t))
    J        - df(t)/dx(t)
    D        - delay operator dx(t)/dt = f(x(t - d)) = D(d)*f(x(t))

    Prior fixed parameter scaling [Defaults]

    M.pF.E = [32 16 4];           % extrinsic rates (forward, backward, lateral)
    M.pF.H = [1 4/5 1/4 1/4]*128; % intrinsic rates (g1, g2 g3, g4)
    M.pF.D = [2 16];              % propogation delays (intrinsic, extrinsic)
    M.pF.G = [4 32];              % receptor densities (excitatory, inhibitory)
    M.pF.T = [8 16];              % synaptic constants (excitatory, inhibitory)
    M.pF.S = [1 1/2];             % parameters of activation function


    JG 2017
    """

    
    
    """
    % get dimensions and configure state variables
    %--------------------------------------------------------------------------   
    n = length(P.A{1});         % number of sources
    x = spm_unvec(x,M.x);       % neuronal states
    """;

    n = len(P['A'][0]);        # % number of sources      # TO DO
    
    
    x = spm_unvec(x,M['x']);       # % neuronal states

    
    """
    #% [default] fixed parameters
    #%--------------------------------------------------------------------------
    E = [1 1/2 1/8]*32;         % extrinsic rates (forward, backward, lateral)
    G = [1 4/5 1/4 1/4]*128;    % intrinsic rates (g1 g2 g3 g4)
    D = [2 16];                 % propogation delays (intrinsic, extrinsic)
    H = [4 32];                 % receptor densities (excitatory, inhibitory)
    T = [8 16];                 % synaptic constants (excitatory, inhibitory)
    R = [2 1]/3;                % parameters of static nonlinearity
    """;
    
    E = np.array([1., 1/2., 1/8.,])*32;         # extrinsic rates (forward, backward, lateral)
    G = np.array([1, 4/5., 1/4., 1/4*128.]);   # % intrinsic rates (g1 g2 g3 g4)    
    D = np.array([2, 16]);  #               % propogation delays (intrinsic, extrinsic)
    H = np.array([4, 32]);  #               % receptor densities (excitatory, inhibitory)
    T = np.array([8, 16]);  #               % synaptic constants (excitatory, inhibitory)
    R = np.array([2, 1])/3.;                #% parameters of static nonlinearity

    
    """
    % [specified] fixed parameters
    %--------------------------------------------------------------------------
    if isfield(M,'pF')
     try, E = M.pF.E; end
     try, G = M.pF.H; end
     try, D = M.pF.D; end
     try, H = M.pF.G; end
     try, T = M.pF.T; end
     try, R = M.pF.R; end
    end   
    """   
    
    
    if 'pF' in M:             # if isfield(M,'pF')
        try: E = M['pF']['E']; 
        except: _
        try: G = M['pF']['H']; 
        except: _
        try: D = M['pF']['D']; 
        except: _
        try: H = M['pF']['G']; 
        except: _
        try: T = M['pF']['T']; 
        except: _  
        try: R = M['pF']['R']; 
        except: _  

    """
    #% test for free parameters on intrinsic connections
    #%--------------------------------------------------------------------------
    try
      G = G.*exp(P.H);
    end
    G     = ones(n,1)*G;
    """
    try:
        G = G*np.exp(P['H']);
    except: _
    G     = np.ones([n,1])*G;
    
    
    """
    % exponential transform to ensure positivity constraints
    %--------------------------------------------------------------------------
    A{1}  = exp(P.A{1})*E(1);
    A{2}  = exp(P.A{2})*E(2);
    A{3}  = exp(P.A{3})*E(3);
    C     = exp(P.C);
    """
    A[0] = np.exp(P['A'][0])*E[0]
    A[1] = np.exp(P['A'][1])*E[1]    
    A[2] = np.exp(P['A'][2])*E[2]
    C = np.exp(P['C'])
    
    """
    % intrinsic connectivity and parameters
    %--------------------------------------------------------------------------
    Te    = T(1)/1000*exp(P.T(:,1));         % excitatory time constants
    Ti    = T(2)/1000*exp(P.T(:,2));         % inhibitory time constants
    He    = H(1)*exp(P.G(:,1));              % excitatory receptor density
    Hi    = H(2)*exp(P.G(:,2));              % inhibitory receptor density
    """
    Te = T[0]/1000*np.exp(P['T'][:,0])
    Ti = T[1]/1000*np.exp(P['T'][:,1])
    He = H[0]*np.exp(P['G'][:,0])
    Hi = H[1]*np.exp(P['G'][:,1])    

    """
    % pre-synaptic inputs: s(V)
    %--------------------------------------------------------------------------
    R     = R.*exp(P.S);
    S     = 1./(1 + exp(-R(1)*(x - R(2)))) - 1./(1 + exp(R(1)*R(2)));
    """
    R = R*np.exp(P['S'])
    S = 1./(1 + np.exp(-R[0]*(x - R[1]))) - 1./(1 + np.exp(R[0]*R[1]));
    
    """
    % input
    %==========================================================================
    if isfield(M,'u')
    
      % endogenous input
      %----------------------------------------------------------------------
      U = u(:)*64;
    
    else
      % exogenous input
      %----------------------------------------------------------------------
      U = C*u(:)*2;
    end
    """
    if 'u' in M:
        #% endogenous input
        #%----------------------------------------------------------------------
        #U = u(:)*64;
        U = u[:]*64.
        
    else:
        #% exogenous input
        #%----------------------------------------------------------------------
        #U = C*u(:)*2;
        U = C*u[:]*2
    
    
    #% State: f(x)
    #%==========================================================================

    """
    % Supragranular layer (inhibitory interneurons): Voltage & depolarizing current
    %--------------------------------------------------------------------------
    f(:,7) = x(:,8);
    f(:,8) = (He.*((A{2} + A{3})*S(:,9) + G(:,3).*S(:,9)) - 2*x(:,8) - x(:,7)./Te)./Te;
    """
    f[:,6] = x[:,7]
    f[:,7] = (He*( (A[1] + A[2]) * S[:,8] + G[:,2]*S[:,8]) - 2*x[:,7] - x[:,6]/Te)/Te
    
    """
    % Granular layer (spiny stellate cells): Voltage & depolarizing current
    %--------------------------------------------------------------------------
    f(:,1) = x(:,4);
    f(:,4) = (He.*((A{1} + A{3})*S(:,9) + G(:,1).*S(:,9) + U) - 2*x(:,4) - x(:,1)./Te)./Te;
    """
    f[:,0] = x[:,3];
    f[:,3] = (He*((A[0] + A[2])*S[:,8] + G[:,0]*S[:,8] + U) - 2*x[:,3] - x[:,0]/Te)/Te;

    """
    % Infra-granular layer (pyramidal cells): depolarizing current
    %--------------------------------------------------------------------------
    f(:,2) = x(:,5);
    f(:,5) = (He.*((A{2} + A{3})*S(:,9) + G(:,2).*S(:,1)) - 2*x(:,5) - x(:,2)./Te)./Te;
    """
    f[:,1] = x[:,4];
    f[:,4] = (He*((A[1] + A[2])*S[:,8] + G[:,1]*S[:,0]) - 2*x[:,4] - x[:,1]/Te)/Te;

    
    """
    % Infra-granular layer (pyramidal cells): hyperpolarizing current
    %--------------------------------------------------------------------------
    f(:,3) = x(:,6);
    f(:,6) = (Hi.*G(:,4).*S(:,7) - 2*x(:,6) - x(:,3)./Ti)./Ti;
    """
    f[:,2] = x[:,5];
    f[:,5] = (Hi*G[:,3]*S[:,6] - 2*x[:,5] - x[:,2]/Ti)/Ti;

    """
    % Infra-granular layer (pyramidal cells): Voltage
    %--------------------------------------------------------------------------
    f(:,9) = x(:,5) - x(:,6);
    f      = spm_vec(f);
    """
    f[:,8] = x[:,4] - x[:,5];
    
    
    #f = spm_vec(f);
    f = spm_vec(f)


    #if nargout < 2; return, end
    if returnJ == False and returnD == False:
        return f
    else: 
    
        """
        % Jacobian
        %==========================================================================
        J  = spm_diff(M.f,x,u,P,M,1);
        """

        # TO DO
        #J  = spm_diff(M.f,x,u,P,M,1);
        J = diff_dfdx(M['f'],x,u,P,M,1)
    
        """
        % delays
        %==========================================================================
        % Delay differential equations can be integrated efficiently (but
        % approximately) by absorbing the delay operator into the Jacobian 
        %
        %    dx(t)/dt     = f(x(t - d))
        %                 = Q(d)f(x(t)) 
        %
        %    J(d)         = Q(d)df/dx
        %--------------------------------------------------------------------------
        De = D(2).*exp(P.D)/1000;
        Di = D(1)/1000;
        De = (1 - speye(n,n)).*De;
        Di = (1 - speye(9,9)).*Di;
        De = kron(ones(9,9),De);
        Di = kron(Di,speye(n,n));
        D  = Di + De;
        """
    
        De = D[1] * np.exp(P['D'])/1000.
        Di = D[0] / 1000.
    
        De = np.array*([1-sps.eye(n,n)])*De
        Di = np.array*([1-sps.eye(9,9)])*Di

        De = np.kron(np.ones([9,9]),De)
        Di = np.kron(Di,sps.eye([n,n]))
                  
        D = Di + De
    
        

        """
        % Implement: dx(t)/dt = f(x(t - d)) = inv(1 + D.*dfdx)*f(x(t))
        %                     = Q*f = Q*J*x(t)
        %--------------------------------------------------------------------------
        Q  = inv(speye(length(J)) + D.*J);
        """;
        
        Q = np.linalg.inv(sps.eye(len(J)) + D*J)

        return f,j,D
    






def spm_dfdx(f,f0,dx):
    """
    port of spm_dfdx
    ...which is defined at the bottom of spm_diff.m
    
    """
    
    """
    function dfdx = spm_dfdx(f,f0,dx)
    % cell subtraction
    %--------------------------------------------------------------------------
    """
    
    """
    if iscell(f)
      dfdx  = f;
      for i = 1:length(f(:))
        dfdx{i} = spm_dfdx(f{i},f0{i},dx);
      end  
    elseif isstruct(f)
      dfdx  = (spm_vec(f) - spm_vec(f0))/dx;
    else
      dfdx  = (f - f0)/dx;
    end
    """;
    
    # Check if input is a dict(cell)
    # ...if it is, loop through elements and 
    # re-call this function
    # ...if it isn't, 
    # 

    dfdx = []
    if type(f) == list:
        dfdx  = f;
        for i in f[:]: 
            dfdx[i] = spm_dfdx(f[i],f0[i],dx);
    elif type(f) == dict: 
      dfdx  = (spm_vec(f) - spm_vec(f0))/dx;
    else:
      dfdx  = (f - f0)/dx;
   


def spm_vec(X):
    # Vectorise a numeric, cell or structure array - a compiled routine
    # FORMAT [vX] = spm_vec(X)
    #X  - numeric, cell or stucture array[s]
    #vX - vec(X)
    #X = [np.eye(2),3]
    
    #  Usage:
    #vX = vec(X)
    
    for v_it,v in enumerate(X):
        if v_it == 0:
            try:
                nr,nc = v.shape
                vX = np.reshape(v,[nr*nc,1])
            except:
                vX = np.array([v][:,np.newaxis])
        else:
            try:
                nr,nc = v.shape
                vX = np.concatenate([vX,np.reshape(v,[nr*nc,1])])
            except:
                vX = np.concatenate([vX,np.array([v])[:,np.newaxis]])            
                
    return np.squeeze(vX)[:,np.newaxis]



def spm_unvec(Xflat,X):

    #nr,nc = X.shape
    #vX = np.reshape(X,[nr,nc])    
    
    cnt = 0
    vXlist = []
    for v_it,v in enumerate(X):
        try:
            nr,nc = v.shape
            rng = np.arange(cnt,nr*nc)
            vX = np.reshape(Xflat[rng],[nr,nc])
            vXlist.append(vX)
            cnt+=1
        except: 
            vXlist.append(v)
    
    return vXlist



 




"""

% SLIMMED FUNCTIONS:
% =================

""";


"""
function [y,pst] = slim_gen_erp(P,M,U)
    % Generates a prediction of trial-specific source activity
"""
def slim_gen_erp(P,M,U):

    """
    % check input u = f(t,P,M) and switch off full delay operator
    %--------------------------------------------------------------------------
    try, M.fu; catch, M.fu  = 'spm_erp_u'; end
    try, M.ns; catch, M.ns  = 128;         end
    try, M.N;  catch, M.N   = 0;           end
    try, U.dt; catch, U.dt  = 0.004;       end
    """
    if 'fu' not in M:  M['fu'] = 'spm_erp_u'
    if 'ns' not in M: M['ns'] = 128
    if 'N' not in M: M['N'] = 0
    if 'dt' not in U: U['dt'] = 0.004 


    """
    % peristimulus time
    %--------------------------------------------------------------------------
    % if nargout > 1
    %     pst = (1:M.ns)*U.dt - M.ons/1000;
    % end
    """
    pst = np.arrange(1,M['ns'])*U['dt'] - M['ons']/1000.



    """
    % within-trial (exogenous) inputs
    %==========================================================================
    %if ~isfield(U,'u')
  
    % peri-stimulus time inputs
    %----------------------------------------------------------------------
    U.u = feval(M.fu,(1:M.ns)*U.dt,P,M);
    """
    U['u'] = M['fu'](np.arange(1,M['ns'])*U['dt'],P,M)
        
    #% end
    
    """

    % if isfield(M,'u')
    
    % remove M.u to preclude endogenous input
    %----------------------------------------------------------------------
    %     M = rmfield(M,'u');
    
    % end
    """
    # (commented; don't need to do)

    """  
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
    """
    # commented, don't need to do)

    """    
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
    """
    y = {}
    for c in np.arange(0,X.shape()[0]):
      Q = slim_gen_Q(P, X[c,:])
      M['x'] = slim_dcm_neural_x(Q,M)
      y[c] = slim_int_L(Q,M,U)




"""    
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
"""

def slim_gen_Q(P,X):
    Q = P['B'] # ? 
    for i in np.arange(0,len(X)):
      for j in np.arange(0, len(Q['A'])):
        Q['A']['j'] = Q['A'] + X[i]*P['B'][i]
   
      Q['G'][:,0] = Q['G'][:,0] + X[i]*np.diag(P['B'][i])

    return Q


"""
function [x] = slim_dcm_neural_x(P,M)
%% Returns the fixed point or steady-state of a neural mass DCM (only for cnductance based models)
x    = M.x;
"""

def slim_dcm_neural_x(P,M):
  x = M['x']

  return x



"""
function [y] = slim_int_L(P,M,U)
    % Integrate a MIMO nonlinear system using a fixed Jacobian: J(x(0))
    % FORMAT [y] = spm_int_L(P,M,U)
    % P   - model parameters
    % M   - model structure
    % U   - input structure or matrix
    %
    % y   - (v x l)  response y = g(x,u,P)
"""

def slim_int_L(P,M,U):
    """    
    dt = U.dt;
    """
    dt = U['dt']

    """
    % Initial states and inputs
    x   = M.x;
    u   = U.u(1,:);
    """
    x = M['x']
    u = U['u'][0,:]
 
    """
    % add [0] states if not specified
    f   = str2func(M.f);
    M.f = f;
    """
    f = M['f']   # completely unnecessary...
    M['f'] = f  


    """
    % output nonlinearity, if specified
    
    g = @(x,u,P,M) x;
    M.g = g;
    """
    M['g'] = g

   
    """
    % dx(t)/dt and Jacobian df/dx and check for delay operator
    %--------------------------------------------------------------------------
    D       = 1;
    n       = numel(x);
    [fx,dfdx,D] = f(x,u,P,M);
       
    OPT.tol = 1e-6*norm((dfdx),'inf');
    
    p = abs(eigs(dfdx,1,'SR',OPT));
    
    N     = ceil(max(1,dt*p*2));
    Q     = (spm_expm(dt*D*dfdx/N) - speye(n,n))*spm_inv(dfdx);
    """
    D = 1
    n = len(x)
    fx,dfdx,D = f(x,u,P,M)
    
    OPT = {'tol': 1E-6*np.norm(dfdx,'inf') }
   
    p = np.abs(np.linalg.eig(df,1,'SR',OPT))

    N = np.ceil(np.max(1,dt*p*2))
    Q = (spm_expm(dt*D*dfdx/N) - np.speye(n,n))*spm_inv(dfdx)


    """
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
    """
    v = spm_vec(x)
    for i in np.arange(0,U.u.shape[0],1):
      u = U['u'][i,:]
      for j in np.arange(0,N):
        v = v+Q*f(v,u,P,M)
      y[:,i] = g(v,u,P,M)

    """
    % transpose
    %--------------------------------------------------------------------------
    y      = real(y');
    """
    y = np.real(y.T)
 
"""
function [x,f,h] = slim_dcm_x_neural(P,model)
    % Returns the state and equation of neural mass models - case{'erp'}
    % linear David et al model (linear in states)
    disp(model)
    n  = length(P.A{1});                          % number of sources
    m  = 9;                                       % number of states
    x  = sparse(n,m);
    f  = 'slim_fx_erp';
"""
def slim_dcm_x_neural(P,model):
   n = len(P['A'][0])
   m = 9
   x = sparse(n,m)
   f = 'slim_fx_erp'
   h = _ # ??
   return x,f,h
 
"""
function [pE,pC] = slim_dcm_neural_priors(A,B,C,model)
    % Prepares the priors on the parameters of neural mass models
    %     case{'erp','sep'}
    [pE,pC] = slim_erp_priors(A,B,C);
"""
def slim_dcm_neural_priors(A,B,C,model):
  pE,pC = slim_erp_priors(A,B,C)
  return pE,pC


 
"""
function [E,V] = slim_erp_priors(A,B,C)
    % prior moments for a neural-mass model of ERPs
"""
 
def slim_erp_priors(A,B,C):

    """ 
    % default: a single source model
    N     = 4;
    n     = size(C,1);                                % number of sources
    u     = size(C,2);                                % number of inputs
    """
    N = 4
    n = C[0].shape[0]
    u = C[0].shape[1]

    """
    % parameters for neural-mass forward model
    %==========================================================================
    
    % set intrinsic [excitatory] time constants and gain
    %--------------------------------------------------------------------------
    E.T   = sparse(n,2);  V.T = sparse(n,2) + 1/16;   % time constants
    E.G   = sparse(n,2);  V.G = sparse(n,2) + 1/16;   % synaptic density
    """
    # not using sparse matrices for now...
    E,V = {},{}
    E['T'] = np.zeros([n,2])
    E['G'] = np.zeros([n,2])
    V['T'] = np.zeros([n,2]) + 1./16
    V['G'] = np.zeros([n,2]) + 1./16


    """
    % set parameter of activation function
    %--------------------------------------------------------------------------
    E.S   = [0 0];        V.S = [1 1]/16;             % dispersion & threshold
    """
    E['S'] = np.array([0,0]); V['S'] = np.array([1,1])/16.
    
    """
    % set extrinsic connectivity
    %--------------------------------------------------------------------------
    Q     = sparse(n,n);
    for i = 1:length(A)
        A{i} = ~~A{i};
        E.A{i} = A{i}*N - N;                          % forward
        V.A{i} = A{i}/16;                             % backward
        Q      = Q | A{i};                            % and lateral connections
    end
    """
    E['A'],V['A'] = {},{}
    Q = np.zeros([n,n])
    for i in np.arange(0,len(A)):
      A[i] = (np.abs(A[i]) > 0).astype(float)
      E['A'][i] = A[i]*N - N
      V['A'][i] = A[i]/16.
      Q = (Q.astype(bool) + (A[i].astype(bool))).astype(float)

    """
    for i = 1:length(B)
        B{i} = ~~B{i};
        E.B{i} = 0*B{i};                              % input-dependent scaling
        V.B{i} = B{i}/8;
        Q      = Q | B{i};
    end
    C      = ~~C;
    E.C    = C*N - N;                                 % where inputs enter
    V.C    = C/32;
    """
    E['B'],E['C'],V['C'],V['B'] = {},{},{},{}
    for i in np.arange(0,len(B)):
      B[i] = (np.abs(B[i]) > 0).astype(float)
      E['B'][i] = 0*B[i]
      V['B'][i] = B[i]/8.
      Q = (Q.astype(bool) + (B[i].astype(bool))).astype(float)
    C[0] = (np.abs(C[0]) > 0).astype(float)
    E['C'] = C[0]*N - N
    V['C'] = C[0]/32.
    


    """
    % set intrinsic connectivity
    %--------------------------------------------------------------------------
    E.H    = sparse(1,4);
    V.H    = sparse(1,4) + 1/16;
    """
    E['H']  = sparse(1,4)
    V['H']  = sparse(1,4) + 1./16.


    """
    % set (extrinsic) delay
    %--------------------------------------------------------------------------
    E.D    = sparse(n,n);
    V.D    = Q/16;
    """
    E['D'] = sparse(n,n)
    V['D'] = Q/16.    


    """ 
    % fix intrinsic delays
    %--------------------------------------------------------------------------
    V.D    = V.D - diag(diag(V.D));
    """    
    V['D'] = V['D'] *np.diag(np.diag(V['D']))


    """
    % set stimulus parameters: onset, dispersion and sustained proportion
    %--------------------------------------------------------------------------
    E.R    = sparse(u,2);  V.R   = E.R + 1/16;
    return
    """
    E['R'] = sparse(u,2)
    V['R'] = E['R'] + 1/16.
    


"""
function [pE,pC] = slim_L_priors(dipfit)
    % prior moments for the lead-field parameters of ERP models
    % defaults
"""

def slim_L_priors(dipfit):
    """
    %--------------------------------------------------------------------------
    model    = dipfit.model;
    type     = dipfit.type;     %ok
    location = 0;
    pC       = [];
    """
    model = dipfit['model']
    type = dipfit['type'] 
    location = 0
    pC = []

    """
    % number of sources
    n = dipfit.Ns;
    m = dipfit.Nc;
    """
    n = dipfit['Ns']
    m = dipfit['Nc']


    """ 
    % location priors (4 mm)
    V = 0;
    """
    V = 0



    """
    % parameters for electromagnetic forward model
    %     case{'LFP'}
    pE.Lpos = sparse(3,0);   pC.Lpos = sparse(3,0);    % positions
    pE.L    = ones(1,m);     pC.L    = ones(1,m)*64;   % gains
    """
    pE['Lpos'] = sparse(3,0); pC['Lpos'] = sparse(3,0)
    pE['L'] = np.ones([1,m]); pC['L'] = np.ones([1,m])*64.


    """    
    % contributing states (encoded in J)
    if ischar(model), mod.source = model; model = mod; end
    pE.J = {};
    pC.J = {};
    """
    mod['source'] = model
    model = mod
    pE['J'] = {}
    pC['J'] = {}

    """
    %         case{'ERP','SEP'}
    %--------------------------------------------------------------
    pE.J{end + 1} = sparse(1,9,1,1,9);               % 9 states
    pC.J{end + 1} = sparse(1,[1 7],1/32,1,9);
    pE.J = pE.J{:};
    pC.J = pC.J{:};
    """
    pE['J'].append(sparse(1,9,1,1,9))
    pC['J'].append(sparse(1,[1,7],1/32.,1,9))
    pE['J'] = pE['J']
    pC['J'] = pC['J']


"""
function [Q] = slim_Q(varargin)   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def slim_Q(instuff):
    """
    a = varargin{1}
    n = varargin{2}
    """
    a = instuff[0]
    n = instuff[1]

    """
    if length(varargin) == 2
        [Q] = spm_Q(a,n);
    elseif length(varargin) == 3
        q = varargin{3}
        [Q] = spm_Q(a,n,q)
    end
    """
    if len(instuff) == 2:
      Q = spm_Q(a,n)
    elif len(instuff) == 3:
      q = instuff[2]
      Q = spm_Q(a,n,q)


"""
function [L] = slim_lx_erp(P,dipfit)
    % observer matrix for a neural mass model: y = G*x
    % parameterised lead field times source contribution to ECD
    L = spm_erp_L(P,dipfit);               % lead field per source
    L = kron(P.J,L);                       % lead-field per state
"""
def slim_lx_erp(P,dipfit):
   L = spm_erp_L(P,dipfit)
   L = np.kron(P['J'],L)


"""
function [L] = spm_erp_L(P,dipfit)
    % returns [projected] lead field L as a function of position and moments
    % type of spatial model and modality
    type = dipfit.type;
    %     case{'LFP'}
    m     = length(P.L);
    n = dipfit.Ns;
    L     = sparse(1:m,1:m,P.L,m,n);
"""
def spm_erp_L(P,dipfit):
  _type = dipfit['type']
  m = len(P['L'])
  n = dipfit['Ns']
  L = sparse(np.arange(0,m), np.arange(0,m), P['L'],m,n)
  return L


""" 
function [K] = slim_sqrtm(V)
    [u,s] = spm_svd(V,0);
    s     = sqrt(abs(diag(s)));
    m     = length(s);
    s     = sparse(1:m,1:m,s);
    K     = u*s*u';
"""     

def slim_sqrtm(V):
  u,s = spm_svd(V,0)        
  s = np.sqrt(np.abs(np.diag(s)))
  m = len(s)
  s = np.sparse(np.arange(0,m),np.arange(0,m), s)
  K = u*s*u.T
  return K




  

"""
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


""" 

    
    
if __name__ == '__main__':

    print 'running'

    main()






    



