% problem3.m
%   HW2 - BMED-8813-BHI
function problem3
    
    N = 100;
    alpha = 0.005*N; % scaling, see piazza discussion
    beta = 0.08;
    times = 0:70;
    
    [S,I,R] = sir(alpha,beta,N,times);
    
    plot(times,S,'b-',...
        times,I,'r-',...
        times,R,'g-');
    
    title('Problem 3: SIR Model with alpha=0.005, beta=0.08');
    xlabel('time');
    ylabel('population');
    
end
