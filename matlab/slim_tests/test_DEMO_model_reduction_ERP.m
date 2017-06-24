function [Y] = test_DEMO_model_reduction_ERP
%
% [Y] = test_DEMO_model_reduction_ERP
%
%
% Slightly modeiifed version of DEMO_model_reduction_ERP.m
%
% Returns several variables as outputs, for comparison with 
% with the slimmed down functions. 
%
% JG June 2017
% __________________________________________________________________________
%
%
%
%
% Illustration of (post hoc)the neuronal mass model optimisation
%__________________________________________________________________________
% This demonstration routine illustrates the post-hoc optimisation of
% dynamic causal models for event related responses. To assess performance
% in relation to ground truth, it uses simulated data. We will simulate a
% simple two source model with exogenous input to the first source and
% reciprocal (extrinsic) connections between the two sources. the ERPs are
% simulated and two conditions, where the second condition induces a change
% in the intrinsic coupling of the first source and the forward extrinsic
% coupling. We then explore a simple model space; created by increasing the
% precision of shrinkage priors on the intrinsic condition specific effect.
% Because this effect was responsible for generating the data, we expect
% the free energy (log evidence) to fall as the shrinkage covariance falls
% to 0). Crucially, we compare and contrast the estimates of the free
% energy (and parameter estimates) using an explicit inversion of the
% reduced models (with tighter shrinkage priors) and a post-hoc model
% reduction procedure – that is computationally more efficient and
% robust to local minima.
%__________________________________________________________________________
% Copyright (C) 2010 Wellcome Trust Centre for Neuroimaging
 
% Karl Friston
% $Id: DEMO_model_reduction_ERP.m 5790 2013-12-08 14:42:01Z karl $

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

% sspecify connectivity – reciprocal connections with condition specific
% changes in intrinsic and extrinsic connectivity
%--------------------------------------------------------------------------
A{1}    = [0 0; 1 0];
A{2}    = [0 1; 0 0];
A{3}    = [0 0; 0 0];
B{1}    = [1 0; 1 0];
C       = [1; 0];

[pE,pC] = spm_dcm_neural_priors(A,B,C,options.model);
[gE,gC] = spm_L_priors(M.dipfit);
[x,f]   = spm_dcm_x_neural(pE,options.model);

% hyperpriors (assuming a high signal to noise)
%--------------------------------------------------------------------------
hE      = 6;
hC      = 1/128;

% create model
%--------------------------------------------------------------------------
M.IS   = 'spm_gen_erp';
M.G    = 'spm_lx_erp';
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
x     = spm_gen_erp(P,M,U);                 % neuronal response
L     = spm_lx_erp(G,M.dipfit);             % lead field
V     = spm_sqrtm(spm_Q(1/2,M.ns));         % square root of noise covariance
for i = 1:length(x)
    n    = exp(-hE/2)*V*randn(M.ns,Nc);     % noise
    s{i} = x{i}*L';                         % signal
    y{i} = s{i} + n;                        % data (signal plus noise)
end

% data structure specification
%--------------------------------------------------------------------------
Y.y   = y;
Y.Q   = {spm_Q(1/2,M.ns,1)};
Y.dt  = dt;
Y.pst = pst;



% (SECOND HALF OF DEMO REMOVED FOR NOW)


