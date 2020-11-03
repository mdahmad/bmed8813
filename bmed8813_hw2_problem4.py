#! /root/anaconda3/bin/python3

##############################
'''
Maria Ahmad
BMED 8813
Homework 2
Problem 3
'''
##############################




##############################
import matplotlib
import matplotlib.pyplot as plt
import scipy

from scipy.integrate import odeint
import numpy as np
from scipy.optimize import minimize, rosen, rosen_der
##############################




##############################
def main():
    function1_result = function1()
    # function2(function1_result)
    return
##############################




##############################
def function1():
    ### Initial conditions
    # Population size
    N = 330000000
    # Initial susceptible population size
    S0 = 329999999
    # Initial infected population size
    I0 = 1
    # Initial recovered populaztion size
    R0 = 0
    # Initial y values
    y0 = S0, I0, R0

    # Initial days list
    days_list = []
    # Initial infected cases list
    infected_cases_list = []
    # Initial susceptible people list
    susceptible_people_list = []

    # Open CSV file
    fileHandle = open('./hw2-covid-usa-data.csv','r')
    for line in fileHandle:
        if 'Cases' not in line:
            # Parse the lines
            line = line.strip().split(',')
            day = line[0]
            infected_number = line[2]
            
            # Calculate susceptible people
            susceptible_number = N - int(infected_number)
            # Add to the lists
            days_list.append(day)
            infected_cases_list.append(infected_number)
            susceptible_people_list.append(susceptible_number)
    fileHandle.close()


    res = minimize(rosen, infected_cases_list, method='Nelder-Mead')
    print(res)
    print('\n\n\n')
    print(res.x)
    print('\n\n\n')
    print(res.final_simplex)
    return res.x

##############################
   



##############################
def deriv(y, t, N, alpha, beta):
    # Set S,I,R equal to the y values
    S, I, R = y
    # Formulas
    dS_dt = (-1) * alpha * S * I
    dI_dt = (alpha * S * I) - (beta * I)
    dR_dt = (beta * I)
    return dS_dt, dI_dt, dR_dt
##############################




##############################
def function2(function1):
    # Population size
    N = 330000000
    # Transmission parameter
    alpha = function1
    # Removal rate parameter
    beta = function1
    # Initial S,I,R values
    S0, I0, R0 = 329999999, 1, 0
    # Days to run the simulation
    days = 121
    # Set up the time space
    t = np.linspace(0,days,days)
    # Initial y values
    y0 = S0, I0, R0
    # Integrate over the time space
    ret = odeint(deriv, y0, t, args=(N,alpha,beta))
    S, I, R = ret.T

    # Initialize the plot
    fix, ax = plt.subplots()
    # Plot the S,I,R values
    plt.plot(list(range(0,days)), S, 'green',label='Susceptible')
    plt.plot(list(range(0,days)), I, 'purple',label='Infected')
    plt.plot(list(range(0,days)), R, 'cyan',label='Recovered')
    # Add labels
    ax.legend()
    plt.title("SIR Model")
    ax.set_ylabel('# Inidividuals')
    ax.set_xlabel('Time (days)')
    # Save figure
    plt.savefig('problem4.png')

    return




main()
