import math

import numpy as np
import scipy
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator
#import random
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
    if W_t == 0:
        return 0
    elif W_t == 1:
        return 0.15
    elif W_t == 2:
        return 0.35
    elif W_t == 3:
        return 0.5
    elif W_t == 4:
        return 0.75
    elif W_t == 5:
        return 1
    elif W_t == 6:
        return 1.25
    elif W_t == 7:
        return 1.4
    elif W_t == 8:
        return 1.45
    elif W_t == 9:
        return 1.48
    elif W_t == 10:
        return 1.5


def R(Q_t,P_t,e_t):
    if P_t >= 0 and Q_t < e_t:
        return K_p_plus*P_t*(e_t-Q_t)
    elif P_t >= 0 and Q_t >= e_t:
        return K_n_plus*P_t*(Q_t - e_t)
    elif P_t < 0 and Q_t < e_t:
        return K_n_minus*P_t*(e_t - Q_t)
    elif P_t < 0 and Q_t >= e_t:
        return K_n_minus*P_t*(Q_t - e_t)


def s_hat(S_t,Q_t,fW_t):
    if Q_t/tau >= fW_t >= 0:
        return min(S_t,C_D,(Q_t/tau-fW_t)/gamma)
    elif fW_t >= Q_t/tau >= 0:
        return - min(C_S-S_t,C_C, (fW_t-Q_t/tau)/theta)
    elif fW_t >= 0 > tau*Q_t:
        return -min(C_S-S_t,C_C,(fW_t-tau*Q_t)*theta)

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

Reward_list = []
for i in range(len(P)):
    for j in range(len(W)):
        for k in range(len(Q)):
            for l in range(len(S)):
                fW_t = f(W[j])
                Reward_list.append(R(Q[k],P[i],E(Q[k],S[l],fW_t)))
max_reward = max(Reward_list)
min_reward = min(Reward_list)
print("Max reward:", max_reward)
print("Min reward:", min_reward)
#-----------------------------------------------------------
T = 168

S_1 = find_nearest(S,C_S/2)
Q_1 = 0
xi_1 = 5
rho_1 = 0
j_1 = 0

#def VIA_risk_neutral(S,Q,W,P,eps=1,max_iter=40):
#    U = np.zeros((len(S),len(Q),len(W),len(P)))
#    Y = np.zeros_like(U)
#    PI = np.zeros_like(U)
#    M_ns = []
#    m_ns = []
#    t = 0
#    while True:
#        t += 1
#        start_time = time.time()
#        for s in range(len(S)):
#            for q in range(len(Q)):
#                for w in range(len(W)):
#                    fW_t = f(W[w])
#                    shat = s_hat(S[s], Q[q], fW_t)
#                    s_ny = find_nearest(S,s+shat)
#                    s_ny_idx = int(np.where(S==s_ny)[0][0])
#                    ehat = E(Q[q], S[s], fW_t)
#                    for p in range(len(P)):
#                        def V(pi):
#                            V_plus1 = 0
#                            for p_ in range(len(P)):
#                                for w_ in range(len(W)):
#                                    V_plus1 += U[s_ny_idx,pi,w_,p_]*P_trans_wind[w][w_]*P_trans_price[p][p_]
#                            return R(Q[q],P[p],ehat)+V_plus1
#                        values = [V(pi) for pi in range(len(Q))]
#                        Y[s,q,w,p] = max(values)
#                        PI[s,q,w,p] = Q[np.argmax(values)]
#        end_time = time.time()
#        print(t, "max", np.max(Y - U), "min", np.min(Y - U), "Running time:", end_time-start_time)
#        M_ns.append(np.max(Y - U))
#        m_ns.append(np.min(Y - U))
#        if np.max(Y - U) - np.min(Y - U) < eps*np.min(Y - U):
#            break
#        if t > max_iter:
#            break
#        U = np.copy(Y)
#
#    return PI, U, M_ns, m_ns

def VIA_risk_neutral(S, Q, W, P, discount = 0.8, eps=1/10, max_iterations=40):
    len_S, len_Q, len_W, len_P = len(S), len(Q), len(W), len(P)
    U = np.zeros((len_S, len_Q, len_W, len_P))
    Y = np.zeros_like(U)
    PI = np.zeros_like(U)
    M_ns = []
    m_ns = []
    t = 0

    while True:
        t += 1
        start_time = time.time()

        for s in range(len_S):
            for q in range(len_Q):
                for w in range(len_W):
                    fW_t = f(W[w])
                    shat = s_hat(S[s], Q[q], fW_t)
                    s_ny = find_nearest(S, S[s] + shat)
                    s_ny_idx = int(np.where(S == s_ny)[0][0])
                    ehat = E(Q[q], S[s], fW_t)
                    for p in range(len_P):
                        def V(pi):
                            V_plus1 = np.sum(
                                U[s_ny_idx, pi, :, :]* P_trans_wind[w, :, None] * P_trans_price[p,None, :]
                            )
                            return R(Q[q], P[p], ehat) + discount*V_plus1
                        values = [V(pi) for pi in range(len_Q)]
                        Y[s, q, w, p] = max(values)
                        PI[s, q, w, p] = Q[np.argmax(values)]

        end_time = time.time()
        print(
            t,
            "max",
            np.max(Y - U),
            "min",
            np.min(Y - U),
            "Running time:",
            end_time - start_time,
        )
        M_ns.append(np.max(Y - U))
        m_ns.append(np.min(Y - U))

        if np.max(Y - U) - np.min(Y - U) < eps * np.min(Y - U):
            print("Converged")
            break
        if t > max_iterations:
            print("Max iterations reached")
            break

        U = np.copy(Y)

    return PI, U, M_ns, m_ns


#----------------------------------------------------------------


#----------------------------------------------------------------
# Risk preferences

def VIA_risk_entropic(S, Q, W, P, discount = 0.95,  eps = 1/10, max_iterations=40, risk_param = 0.8):
    len_S, len_Q, len_W, len_P = len(S), len(Q), len(W), len(P)
    U = np.zeros((len_S, len_Q, len_W, len_P))
    Y = np.zeros_like(U)
    PI = np.zeros_like(U)
    M_ns = []
    m_ns = []
    t = 0

    while True:
        t += 1
        start_time = time.time()

        for s in range(len_S):
            for q in range(len_Q):
                for w in range(len_W):
                    fW_t = f(W[w])
                    shat = s_hat(S[s], Q[q], fW_t)
                    s_ny = find_nearest(S, S[s] + shat)
                    s_ny_idx = int(np.where(S == s_ny)[0][0])
                    ehat = E(Q[q], S[s], fW_t)
                    for p in range(len_P):
                        def V(pi):
                            V_plus1 = np.sum(
                                np.exp(-risk_param * (U[s_ny_idx, pi, :, :])) * P_trans_wind[w, :, None] * P_trans_price[p, None, :]
                            )
                            if math.isnan(V_plus1) == True:
                                print(-risk_param*U[s_ny_idx, pi, :, :])
                                print(np.exp(-risk_param*U[s_ny_idx, pi, :, :]))
                            return R(Q[q], P[p], ehat)/100 - discount/risk_param*np.log(V_plus1) #deler med 1000 for at undgå overflow
                        values = [V(pi) for pi in range(len_Q)]
                        Y[s, q, w, p] = max(values)
                        PI[s, q, w, p] = Q[np.argmax(values)]

        end_time = time.time()
        print(
            t,
            "max",
            np.max(Y - U),
            "min",
            np.min(Y - U),
            "Running time:",
            end_time - start_time,
        )
        M_ns.append(np.max(Y - U))
        m_ns.append(np.min(Y - U))

        if np.max(Y - U) - np.min(Y - U) < eps * np.min(Y - U):
            print("Converged")
            break
        if t > max_iterations:
            print("Max iterations reached")
            break

        U = np.copy(Y)

    return PI, U, M_ns, m_ns

def VIA_risk_CVaR(S, Q, W, P, discount = 0.95,  eps = 1/10, max_iterations=40, level = 0.8):
    len_S, len_Q, len_W, len_P = len(S), len(Q), len(W), len(P)
    U = np.zeros((len_S, len_Q, len_W, len_P))
    Y = np.zeros_like(U)
    PI = np.zeros_like(U)
    M_ns = []
    m_ns = []
    t = 0

    while True:
        t += 1
        start_time = time.time()

        for s in range(len_S):
            print("hej", s)
            for q in range(len_Q):
                for w in range(len_W):
                    fW_t = f(W[w])
                    shat = s_hat(S[s], Q[q], fW_t)
                    s_ny = find_nearest(S, S[s] + shat)
                    s_ny_idx = int(np.where(S == s_ny)[0][0])
                    ehat = E(Q[q], S[s], fW_t)
                    for p in range(len_P):
                        def V(pi):
                            def CVAR(eta):
                                V_plus1 = np.sum(
                                    np.maximum(U[s_ny_idx, pi, :, :]-eta,0) * P_trans_wind[w, :,
                                                                                    None] * P_trans_price[p, None, :]
                                )
                                return eta - 1/(1-level)*V_plus1
                            res = scipy.optimize.minimize(CVAR, 0, bounds = ((min_reward,max_reward),))
                            return R(Q[q], P[p], ehat)/100 + discount*res.fun
                        values = [V(pi) for pi in range(len_Q)]
                        Y[s, q, w, p] = max(values)
                        PI[s, q, w, p] = Q[np.argmax(values)]

        end_time = time.time()
        print(
            t,
            "max",
            np.max(Y - U),
            "min",
            np.min(Y - U),
            "Running time:",
            end_time - start_time,
        )
        M_ns.append(np.max(Y - U))
        m_ns.append(np.min(Y - U))

        if np.max(Y - U) - np.min(Y - U) < eps * np.min(Y - U):
            print("Converged")
            break
        if t > max_iterations:
            print("Max iterations reached")
            break

        U = np.copy(Y)

    return PI, U, M_ns, m_ns



PI, U, M_ns, m_ns = VIA_risk_CVaR(S,Q,W,P,eps=1/10, max_iterations=40)

