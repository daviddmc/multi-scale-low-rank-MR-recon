%% Demo for Multi-scale Low Rank Decomposition on Dynamic-Contrast Enhanced MRI
% 
% (c) Frank Ong 2015
clc
clear
close all
setPath
addpath('nufft_toolbox/');

%% Set Parameters


% number of spokes to be used per frame (Fibonacci number)
nspokes=21;%21
% load radial data
% load abdomen_dce_ga.mat
load liver_data.mat
%load simulated_20.mat

%w = dcf;
[nx, ny, nc] = size(b1);
[nr,ntviews,nc] = size(kdata);

%
%
b1=b1/max(abs(b1(:)));
for ch=1:nc,kdata(:,:,ch)=kdata(:,:,ch).*sqrt(w);end
%
%

% number of frames
nt=floor(ntviews/nspokes);
% crop the data according to the number of spokes per frame
kdata = kdata(:,1:nt*nspokes,:);
k = k(:,1:nt*nspokes);
w = w(:,1:nt*nspokes);
% sort the data into a time-series of undersampled images
for ii=1:nt
    kdatau(:,:,:,ii)=kdata(:,(ii-1)*nspokes+1:ii*nspokes,:);
    ku(:,:,ii)=k(:,(ii-1)*nspokes+1:ii*nspokes);
    wu(:,:,ii)=w(:,(ii-1)*nspokes+1:ii*nspokes);
end
% multicoil NUFFT operator
E = MCNUFFT(ku,wu,b1);

tmp = E' * kdatau;
kdatau = kdatau / max(abs(tmp(:)));

Y_size = [nx, ny, nt];

nIter = 64; % Number of iterations

rho = 30; % ADMM parameter 0.5
skip = 2;
dorandshift = 1; % Set do random shifts

% Plot
%figure,imshow3(abs(Y)),title('Input','FontSize',14);
%drawnow

%% Generate Multiscale block Sizes
L = ceil(max( log2( Y_size(1:2) ) ));

% Generate block sizes
block_sizes = [ min( 2.^(0:skip:L)', Y_size(1)) , min( 2.^(0:skip:L)', Y_size(2)), ones(length((0:skip:L)),1)*Y_size(3) ];
disp('Block sizes');
disp(block_sizes)

levels = size(block_sizes,1);

ms = prod(block_sizes(:,1:2),2);

ns = block_sizes(:,3);

bs = repmat( prod( Y_size(1:2) ), [levels,1]) ./ ms;

% Penalties
lambdas = sqrt(ms) + sqrt(ns) + sqrt( log2( bs .* min( ms, ns ) ) );


%% Initialize Operators
YD_size = [Y_size,levels];
decom_dim = length(Y_size) + 1;

% Get summation operator
A = @(x) (E * (sum( x, decom_dim ))) / sqrt(levels);
AT = @(x) repmat( E' * x, [ones(1,decom_dim-1), levels] )/ sqrt(levels);

%% Iterations:

X_it = AT(kdatau);
%[m, u, flag, iter] = PowerIter( @(x) AT(A(x)), 1e-6, 30, u, 'sym');
%error('end')
AX_it = A(X_it);
Z_it = zeros(size(kdatau));
U_it = zeros(size(kdatau));

k = 0;
K = 1;
rho_k = rho;
alpha = 100;

for it = 1:nIter
    X_it = X_it - (1/alpha) * AT(AX_it - Z_it + U_it);
    
    for l = 1:levels
        XU = X_it(:,:,:,l);
        r = [];
        
        if (dorandshift)
            [XU, r] = randshift( XU );
        end
        
        XU = blockSVT3( XU, block_sizes(l,:), lambdas(l) / rho_k);
        
        
        if (dorandshift)
            XU = randunshift( XU, r );
        end
        
        X_it(:,:,:,l) = XU;
    end
    
    AX_it = A(X_it);
    
    mu = alpha / rho_k;
    Z_it = (AX_it + U_it + mu * kdatau) / (mu + 1);
    
    % Update dual
    U_it = U_it - Z_it + AX_it;
    
    % Update rho
    k = k + 1;
    if (k == K)
        rho_k = rho_k * 2;
        U_it = U_it / 2;
        K = K * 2;
        k = 0;
    end
    
    % Plot
    figure(24),
    imshow3(abs(X_it), []),
    titlef(it);
    drawnow
   
end

%{
for it = 1:nIter
    % Data consistency
    X_it = 0.000001 * AT( kdatau - A( Z_it - U_it ) ) + (Z_it - U_it);
    
    % Level-wise block threshold
    for l = 1:levels
        XU = X_it(:,:,:,l) + U_it(:,:,:,l);
        r = [];
        
        if (dorandshift)
            [XU, r] = randshift( XU );
        end
        
        XU = blockSVT3( XU, block_sizes(l,:), lambdas(l) / rho_k);
        
        
        if (dorandshift)
            XU = randunshift( XU, r );
        end
        
        Z_it(:,:,:,l) = XU;
    end
    
    % Update dual
    U_it = U_it - Z_it + X_it;
    
    % Update rho
    k = k + 1;
    if (k == K)
        rho_k = rho_k * 2;
        U_it = U_it / 2;
        K = K * 2;
        k = 0;
    end
    
    % Plot
    figure(24),
    imshow3(abs(X_it), []),
    titlef(it);
    drawnow
end
%}

%% Show Result

figure,imshow4f(abs(X_it)),title('Results','FontSize',14);


