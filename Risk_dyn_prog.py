import math

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import random
import time

#------------------------------------------------
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

#------------------------------------------------------

# Constants for Markov Model
K_p_plus = 0.9
K_n_minus = 0.9
K_p_minus = 1.1
K_n_plus = 0.9
C_S = 400
C_C = 50
C_D = 50
C_T = 200
NPF = 0.0402
eta_a = 25
tau = 0.97

# Constants for electricity price
kappa = 0.357
sigma_p = 15.281
theta = 0.85
gamma = 0.85
P_trans_price = np.array([[0.351,0.585,0.065,0,0],
                          [0.052,0.539,0.409,0,0],
                          [0,0.167,0.667,0.167,0],
                          [0,0,0.409,0.539,0.052],
                          [0,0,0.065,0.585,0.351]])

# Constants for wind speed
gamma_0 = 8.519
gamma_1 = 1.126
gamma_2 = 1.74
omega_1 = 0.002
omega_2 = -32.431
phi = 0.931
sigma_w = 1.558
P_trans_wind = np.array([[0.626,0.206,0.114,0.042,0.010,0.002,0,0,0,0,0],
                         [0.391,0.252,0.201,0.107,0.039,0.009,0.002,0,0,0,0],
                         [0.191,0.217,0.251,0.195,0.101,0.035,0.008,0.001,0,0,0],
                         [0.072,0.133,0.222,0.250,0.188,0.095,0.032,0.007,0.001,0,0],
                         [0.019,0.057,0.139,0.227,0.248,0.182,0.090,0.030,0.007,0.001,0],
                         [0.004,0.018,0.062,0.146,0.231,0.246,0.176,0.084,0.027,0.006,0.001],
                         [0.001,0.004,0.019,0.067,0.153,0.235,0.243,0.169,0.079,0.025,0.006],
                         [0,0.001,0.004,0.021,0.071,0.159,0.239,0.241,0.162,0.075,0.025],
                         [0,0,0.001,0.005,0.024,0.076,0.166,0.243,0.241,0.164,0.081],
                         [0,0,0,0.001,0.006,0.026,0.083,0.177,0.257,0.260,0.191],
                         [0,0,0,0,0.001,0.007,0.031,0.097,0.210,0.317,0.338]])

# State spaces
P = np.array([-52.93,-26.47,0,26.47,52.93])
J = np.linspace(-350,600,20)
W = np.array([0,1,2,3,4,5,6,7,8,9,10])

# Action space
S = np.arange(0,C_S,eta_a)
Q = np.arange(-C_T,tau*C_T,eta_a)

# Defining functions
def f(W_t):
    return 1.5 - math.log(W_t)


def R(Q_t,P_t,W_t,e_t):
    if P_t >= 0 and Q_t < e_t:
        return Q_t*P_t + K_p_plus*P_t*(e_t-Q_t)
    elif P_t >= 0 and Q_t >= e_t:
        return Q_t*P_t - K_n_plus*P_t*(Q_t - e_t)
    elif P_t < 0 and Q_t < e_t:
        return Q_t*P_t + K_n_minus*P_t*(e_t - Q_t)
    elif P_t < 0 and Q_t >= e_t:
        return Q_t*P_t - K_n_minus*P_t*(Q_t - e_t)


def s_hat(S_t,Q_t,fW_t):
    if Q_t/tau >= fW_t >= 0:
        return min(S_t,C_D,(Q_t/tau-fW_t)/gamma)
    elif fW_t >= Q_t/tau >= 0:
        return - min(C_S-S_t,C_C, (fW_t-Q_t/tau)/theta)
    elif fW_t >= 0 > tau*Q_t:
        return -min(C_s-S_t,C_C,(fW_t-tau*Q_t)*theta)

def w_hat(S_t,Q_t,fW_t):
    if Q_t/tau >= fW_t >= 0:
        return fW_t
    elif fW_t >= Q_t/tau >= 0:
        return Q_t/tau - min((C_S-S_t)/theta, C_C/theta, fW_t-Q_t/tau)
    elif fW_t >= 0 > tau*Q_t:
        return tau*Q_t + min((C_S-S_t)/theta, C_C/theta, fW_t - tau*Q_t)

def E(Q_t,S_t,fW_t):
    if Q_t/tau >= fW_t >= 0:
        return (min(gamma*S_t,gamma*C_D,Q_t/tau-fW_t)+fW_t)*tau
    elif fW_t >= Q_t/tau >= 0:
        return Q_t
    elif fW_t >= 0 > tau*Q_t:
        return Q_t


# ---------------------------------------------------------

def price_transition(W_t):
    price_trans = dict()



# ------------------------------------------------------
T = 168

S_1 = find_nearest(S,C_S/2)
Q_1 = 0
xi_1 = 5
rho_1 = 0
j_1 = 0

def VIA(S,Q,W,P,J):
    U = np.zeros(len(S),len(Q),len(W),len(P),len(J))
    Q = np.zeros_like(U)
    for t in reversed(range(T)):


# Value-Iteration Algorithm

