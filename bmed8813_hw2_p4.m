% problem4.m
%   HW2 - BMED-8813-BHI
function problem4
    
    % import COVID data
    data = readmatrix('hw2-covid-usa-data.csv');
    
    times = data(:,1);
    cases = data(:,3);
    
    % fminsearch minimizes an "objective function" - we will define our
    % objective function to be the MSE between the simulation and the true
    % data (cumulative cases, i.e. I+R)
    function err = sir_error(params)
        [S,I,R] = sir(params(1),params(2),330e6,times);
        err = mean(((I+R)-cases).^2);
    end
    
    initial_guess = [1 1]; % some reasonable starting point
    opt = fminsearch(@sir_error, initial_guess);
    
    fprintf('alpha: %f\nbeta: %f\n',opt(1),opt(2));
    
    [S,I,R] = sir(opt(1),opt(2),330e6,times);
    
    plot(times,I,'r-',times,R,'g-');
    hold on
    plot(times,cases,'cx');
    
    title('Problem 4: SIR Model Fit to COVID Data');
    xlabel('time');
    ylabel('population');
    legend('Simulated I','Simulated R','COVID cases','Location','northwest');
    
end

%#ok<*ASGLU>
