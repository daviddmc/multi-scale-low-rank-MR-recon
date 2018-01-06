function [m, u, flag, iter] = PowerIter( A, tol, maxIter, u0, symString)
% PowerIter     Power iteration method.
%   M = PowerIter(A) tries to find the dominant eigenvalue of matrix A.
%
%   M = PowerIter(A, TOL) specifies the tolerance of the method. If 
%   TOL is [] then PowerIter uses the default, 1e-6.
%
%   M = PowerIter(A, TOL, MAXITER) specifies the maximum number of
%   iterations. If MAXITER is [] then PowerIter uses the default, 
%   max(N, 20).
%
%   M = PowerIter(A, TOL, MAXITER, U0) specifies the initial guess of the 
%   eigenvector of the dominant eigenvalue. If U0 is [] then PowerIter uses 
%   the default, an all one vector.
%
%   M = PowerIter(A, TOL, MAXITER, U0, 'symmetric') assumes that A is
%   symmetric(Hermitian) matrix.
%
%   [M, U] = PowerIter(A, ...) also returns the corresponding eigenvector.
%
%   [M, U, FLAG] = PowerIter(A, ...) also returns a convergence FLAG:
%    0 PowerIter converged to the desird tolerance TOL within MAXITER
%      iterations.
%    1 PowerIter iterated MAXITER times but did not converge.
%
%   [M, U, FLAG, ITER] = PowerIter(A, ...) also returns the iteration
%   number at which M was computed: 0 <= ITER <= MAXITER.
%
%   See also InverseIter.

%   Copyright 2017 Junshen Xu

flag = 1;
u = u0;
m0 = 0;
m1 = 0;
if(exist('symString','var') && strncmpi(symString,'sym',1))
    u = u / norm(u(:));
    for iter = 1:maxIter
        disp(['iter' num2str(iter)]);
        v = A(u);
        m = real(u(:)'*v(:));
        vNorm = norm(v(:));
        if(norm(v(:)/vNorm - u(:)) < tol)
            flag = 0;
            if iter>=4
                m = m0 - (m1 - m0)^2 / (m -2*m1 + m0);
            end
            break;
        end
        u = v / vNorm;
        m0 = m1;
        m1 = m;
        disp(['iter' num2str(iter)]);
        disp(['m = ' num2str(m)]);
    end
else
    
end


