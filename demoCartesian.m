%% Demo for Multi-scale Low Rank Decomposition on Dynamic-Contrast Enhanced MRI
% 
% (c) Frank Ong 2015
clc
clear
close all
setPath

%% Set Parameters
load cardiac_perf_R8.mat;
%load cardiac_cine_R6
[nx,ny,nt,nc]=size(kdata);
[E, ET] = EOP(kdata(:,:,:,1) ~= 0, b1);

%Y_size = size(Y); % Matrix Size
Y_size = [nx, ny, nt];

nIter = 80; % Number of iterations

rho = 0.5; % ADMM parameter
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
A = @(x) E(sum( x, decom_dim )) / sqrt(levels);
AT = @(x) repmat( ET(x), [ones(1,decom_dim-1), levels] ) / sqrt(levels);

%% Iterations:

X_it = zeros(YD_size);
Z_it = zeros(YD_size);
U_it = zeros(YD_size);

k = 0;
K = 1;
rho_k = rho;

for it = 1:nIter
    % Data consistency
    X_it = AT( kdata - A( Z_it - U_it ) ) + Z_it - U_it;
    
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

%% Show Result

figure,imshow4f(abs(X_it)),title('Results','FontSize',14);


