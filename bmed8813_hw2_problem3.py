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
##############################




##############################
def main():
    function1()
    return True
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
def function1():
    # Population size
    N = 100
    # Transmission parameter
    alpha = 0.005
    # Removal rate parameter
    beta = 0.08
    # Initial S,I,R values
    S0, I0, R0 = 99, 1, 0
    # Days to run the simulation
    days = 200
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
    plt.savefig('problem3.png')

    return
    
main()
