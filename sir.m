% sir.m
%   HW2 - BMED-4813/8813-BHI
function [S,I,R] = sir(alpha,beta,N,times)
    
    [t,x] = ode45(@(t,x)sir_model(t,x,alpha,beta,N),times,[N-1,1,0]);
    
    S = x(:,1);
    I = x(:,2);
    R = x(:,3);
    
end

% dxdt = sir_model(t,x,alpha,beta)
%   the SIR diffeqs
function dxdt = sir_model(t,x,alpha,beta,N)
    
    dxdt = zeros(3,1);
    
    dxdt(1) = -alpha*x(1)*x(2)/N;
    dxdt(2) = alpha*x(1)*x(2)/N - beta*x(2);
    dxdt(3) = beta*x(2);
    
end

%#ok<*ASGLU,*INUSL>
