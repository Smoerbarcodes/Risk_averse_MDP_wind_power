import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.optimize import curve_fit
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

scal = 100

# Constants for Markov Model
K_p_plus = 0.9
K_n_minus = 0.9
K_p_minus = 1.1
K_n_plus = 1.1
C_S = 200
C_C = 50
C_D = 50
C_T = 200
NPF = 0.0402
eta_a = 25
tau = 0.97
theta = 0.85
gamma = theta

# Constants for electricity price
kappa = 0.357
sigma_p = 15.281
gamma_1_p = 39.7689
P_trans_price = np.array([[0.350,0.585,0.065,0,0],
                          [0.052,0.539,0.409,0,0],
                          [0,0.167,0.666,0.167,0],
                          [0,0,0.409,0.539,0.052],
                          [0,0,0.065,0.585,0.35]]) #Note, that I have changed a few values making the sum to 1

# Constants for wind speed
gamma_0 = 8.519
gamma_1 = 1.126
gamma_2 = 1.74
omega_1 = 0.002
omega_2 = -32.431
phi = 0.931
sigma_w = 1.558
P_trans_wind = np.array([[0.626,0.206,0.114,0.042,0.010,0.002,0,0,0,0,0],
                         [0.390,0.252,0.201,0.107,0.039,0.009,0.002,0,0,0,0],
                         [0.192,0.217,0.251,0.195,0.101,0.035,0.008,0.001,0,0,0],
                         [0.072,0.133,0.222,0.250,0.188,0.095,0.032,0.007,0.001,0,0],
                         [0.019,0.057,0.139,0.227,0.248,0.182,0.090,0.030,0.007,0.001,0],
                         [0.004,0.018,0.062,0.146,0.231,0.245,0.176,0.084,0.027,0.006,0.001],
                         [0.001,0.004,0.019,0.067,0.153,0.234,0.243,0.169,0.079,0.025,0.006],
                         [0,0.001,0.004,0.021,0.071,0.159,0.240,0.242,0.162,0.075,0.025],
                         [0,0,0.001,0.005,0.024,0.076,0.166,0.242,0.241,0.164,0.081],
                         [0,0,0,0.001,0.006,0.026,0.083,0.177,0.256,0.260,0.191],
                         [0,0,0,0,0.001,0.007,0.031,0.097,0.210,0.317,0.337]])


# State spaces
P = np.array([-52.93,-26.47,0,26.47,52.93])
J = np.linspace(-350,600,20)
W = np.array([0,1,2,3,4,5,6,7,8,9,10])


# Action space
S = np.arange(0,C_S+eta_a,eta_a)
Q = np.arange(0,tau*C_T+eta_a,eta_a)
#Q = np.arange(-C_T,tau*C_T+eta_a,eta_a)


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
    if P_t >= 0 and Q_t <= e_t:
        return Q_t*P_t
    elif P_t >= 0 and Q_t > e_t:
        return Q_t*P_t - K_n_plus*P_t*(Q_t - e_t)
    elif P_t < 0 and Q_t <= e_t:
        return Q_t*P_t
    elif P_t < 0 and Q_t > e_t:
        return Q_t*P_t - K_n_minus*P_t*(Q_t - e_t)

def R_scaled(Q_t,P_t,e_t):
    if P_t >= 0 and Q_t <= e_t:
        return Q_t*P_t/scal
    elif P_t >= 0 and Q_t > e_t:
        return (Q_t*P_t - K_n_plus*P_t*(Q_t - e_t))/scal
    elif P_t < 0 and Q_t <= e_t:
        return (Q_t*P_t)/scal
    elif P_t < 0 and Q_t > e_t:
        return (Q_t*P_t - K_n_minus*P_t*(Q_t - e_t))/scal

def R_risky(Q_t,P_t,e_t):
    if P_t >= 0 and Q_t <= e_t:
        return Q_t*P_t
    elif P_t >= 0 and Q_t > e_t:
        if 40 >= Q_t - e_t > 0:
            return Q_t * P_t - K_n_plus * P_t * (Q_t - e_t)
        elif 80 >= Q_t-e_t > 40:
            return Q_t*P_t-2*K_n_plus*P_t*(Q_t - e_t)
        elif 120 >= Q_t-e_t > 80:
            return Q_t*P_t - 4*K_n_plus*P_t*(Q_t - e_t)
        elif Q_t-e_t > 120:
            return Q_t*P_t - 8*K_n_plus*P_t*(Q_t - e_t)
    elif P_t < 0 and Q_t <= e_t:
        return Q_t*P_t
    elif P_t < 0 and Q_t > e_t:
        if 40 >= Q_t - e_t > 0:
            return Q_t*P_t - K_n_minus*P_t*(Q_t - e_t)
        elif 80 >= Q_t-e_t > 40:
            return Q_t*P_t-2*K_n_minus*P_t*(Q_t - e_t)
        elif 120 >= Q_t-e_t > 80:
            return Q_t*P_t - 4*K_n_minus*P_t*(Q_t - e_t)
        elif Q_t-e_t > 120:
            return Q_t*P_t - 8*K_n_minus*P_t*(Q_t - e_t)

def R_risky_scaled(Q_t,P_t,e_t):
    if P_t >= 0 and Q_t <= e_t:
        return Q_t*P_t/scal
    elif P_t >= 0 and Q_t > e_t:
        if 40 >= Q_t - e_t > 0:
            return (Q_t * P_t - K_n_plus * P_t * (Q_t - e_t))/scal
        elif 80 >= Q_t-e_t > 40:
            return (Q_t*P_t-2*K_n_plus*P_t*(Q_t - e_t))/scal
        elif 120 >= Q_t-e_t > 80:
            return (Q_t*P_t - 4*K_n_plus*P_t*(Q_t - e_t))/scal
        elif Q_t-e_t > 120:
            return (Q_t*P_t - 8*K_n_plus*P_t*(Q_t - e_t))/scal
    elif P_t < 0 and Q_t <= e_t:
        return Q_t*P_t/scal
    elif P_t < 0 and Q_t > e_t:
        if 40 >= Q_t - e_t > 0:
            return (Q_t*P_t - K_n_minus*P_t*(Q_t - e_t))/scal
        elif 80 >= Q_t-e_t > 40:
            return (Q_t*P_t-2*K_n_minus*P_t*(Q_t - e_t))/scal
        elif 120 >= Q_t-e_t > 80:
            return (Q_t*P_t - 4*K_n_minus*P_t*(Q_t - e_t))/scal
        elif Q_t-e_t > 120:
            return (Q_t*P_t - 8*K_n_minus*P_t*(Q_t - e_t))/scal


def R_OG(Q_t,P_t,e_t):
    if P_t >= 0 and Q_t < e_t:
        return Q_t*P_t+K_p_plus*P_t*(e_t-Q_t)
    elif P_t >= 0 and Q_t >= e_t:
        return Q_t*P_t-K_n_plus*P_t*(Q_t - e_t)
    elif P_t < 0 and Q_t < e_t:
        return Q_t*P_t+K_n_minus*P_t*(e_t - Q_t)
    elif P_t < 0 and Q_t >= e_t:
        return Q_t*P_t-K_n_minus*P_t*(Q_t - e_t)


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
difs_list = []
for i in range(len(P)):
    for j in range(len(W)):
        for k in range(len(Q)):
            for l in range(len(S)):
                fW_t = f(W[j])
                ehat = E(Q[k],S[l],fW_t)
                Reward_list.append(R_scaled(Q[k],P[i],ehat))
                difs_list.append(ehat-Q[k])
max_reward = max(Reward_list)
min_reward = min(Reward_list)
max_dif = max(difs_list)
min_dif = min(difs_list)
print("Max reward:", max_reward)
print("Min reward:", min_reward)
print("Min dif:", min_dif)
print("Max dif:", max_dif)

#-----------------------------------------------------------
T = 168

S_0 = find_nearest(S,C_S/2)
Q_0 = 0
xi_0 = 5
rho_0 = 0
j_0 = 0

def VIA_risk_neutral(S, Q, W, P, R, discount = 0.99, eps=1/10, max_iterations=40):
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
                    fW_t = f(W[w])*100
                    shat = s_hat(S[s], Q[q], fW_t)
                    s_ny = find_nearest(S, S[s] + shat)
                    s_ny_idx = int(np.where(S == s_ny)[0][0])
                    ehat = E(Q[q], S[s], fW_t)
                    for p in range(len_P):
                        for pi in range(len(Q)):
                            V_next = np.zeros(len_Q)
                            for pi in range(len_Q):
                                V_plus1 = np.sum(
                                    U[s_ny_idx, pi, :, :] * P_trans_wind[w, :, None] * P_trans_price[p, None, :]
                                )
                                V_next[pi] = V_plus1

                            values = np.array([
                                #R(Q[q], P[p] + gamma_1_p, ehat) + discount * V_next[pi]
                                R(Q[q], P[p], ehat) + discount * V_next[pi]
                                for pi in range(len_Q)
                            ])
                        #def V(pi):
                        #    V_plus1 = np.sum(
                        #        U[s_ny_idx, pi, :, :] * P_trans_wind[w, :, None] * P_trans_price[p,None, :]
                        #    )
                        #    return R(Q[q], P[p]+gamma_1_p, ehat) + discount*V_plus1
                        #values = [V(pi) for pi in range(len_Q)]
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

        if np.max(np.abs(Y - U)) <= eps * (1 - discount) / (2 * discount):
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

def VIA_risk_entropic(S, Q, W, P,R, discount = 0.95,  eps = 1/10, max_iterations=20, risk_param = 0.2):
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
                    fW_t = f(W[w])*100
                    shat = s_hat(S[s], Q[q], fW_t)
                    s_ny = find_nearest(S, S[s] + shat)
                    s_ny_idx = int(np.where(S == s_ny)[0][0])
                    ehat = E(Q[q], S[s], fW_t)
                    for p in range(len_P):
                        values = []
                        for pi in range(len_Q):
                            V_next = U[s_ny_idx, pi, :, :].flatten()
                            probs = (P_trans_wind[w, :, None] * P_trans_price[p, None, :]).flatten()

                            # entropic risk mapping
                            risk_value = (1.0 / -risk_param ) * logsumexp(
                                -risk_param * V_next, b=probs
                            )
                            val = R(Q[q], P[p], ehat) + discount * risk_value
                            values.append(val)
                        #def V(pi):
                        #    V_plus1 = np.sum(
                        #        np.exp(-risk_param * (U[s_ny_idx, pi, :, :])) * P_trans_wind[w, :, None] * P_trans_price[p, None, :][0]
                        #    )
                        #
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

        if np.max(np.abs(Y-U)) <= eps*(1-discount)/(2*discount):
            print("Converged")
            break
        if t > max_iterations:
            print("Max iterations reached")
            break

        U = np.copy(Y)

    return PI, U, M_ns, m_ns


def cvar_from_sorted(values, probs, alpha):
    sorted_pairs = sorted(zip(values, probs), key=lambda x: x[0])
    cumulative = 0
    cvar_sum = 0
    tail_mass = 1 - alpha
    for z, p in sorted_pairs:
        if cumulative + p <= tail_mass:
            cvar_sum += z * p
            cumulative += p
        else:
            remaining = tail_mass - cumulative
            if remaining > 0:
                cvar_sum += z * remaining
            break
    return cvar_sum / tail_mass

def VIA_risk_CVaR_dual(S, Q, W, P,R, discount = 0.95,  eps = 2, max_iterations=40, level = 0.8):
    assert 0 < level < 1, "level must be between 0 and 1"
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
                    fW_t = f(W[w])*100
                    shat = s_hat(S[s], Q[q], fW_t)
                    s_ny = find_nearest(S, S[s] + shat)
                    s_ny_idx = int(np.where(S == s_ny)[0][0])
                    ehat = E(Q[q], S[s], fW_t)
                    for p in range(len_P):
                        def V(pi):
                            #frac_level = level * len_W * len_P
                            #l_level = int(math.ceil(frac_level) - 1)
                            #beta = frac_level - l_level
                            #asc_val = np.sort(U[s_ny_idx, pi, :, :], axis=None)
                            #quantile = asc_val[l_level]
                            #weights = np.where(U[s_ny_idx, pi, :, :]<=quantile, 1/level, 0)
                            #if np.max(U[s_ny_idx, pi, :, :])-np.min(U[s_ny_idx, pi, :, :]) < 1/100:
                            #    weights = np.zeros_like(U[s_ny_idx, pi, :, :])+1
                            #if t > 1:
                            #    print(t, quantile, level)
                            #    print(pi,U[s_ny_idx, pi, :, :])
                            #    print(pi,weights)
                            #V_plus1 = np.sum(np.multiply(U[s_ny_idx, pi, :, :],weights) * P_trans_wind[w, :, None] * P_trans_price[p,None, :])
                            future_values = U[s_ny_idx, pi, :, :].flatten()
                            probs = (P_trans_wind[w, :, None] * P_trans_price[p, None, :]).flatten()
                            V_plus1 = cvar_from_sorted(future_values, probs, level)
                            return R(Q[q], P[p], ehat) + discount*V_plus1
                        values = [V(pi) for pi in range(len_Q)]
                        #print(t,np.argmax(values))
                        Y[s, q, w, p] = np.max(values)
                        if t == 1:
                            PI[s, q, w, p] = Q[np.random.choice(len_Q)]
                        else:
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

        if np.max(np.abs(Y-U)) <= eps*(1-discount)/(2*discount):
            print("Converged")
            break
        if t > max_iterations:
            print("Max iterations reached")
            break

        U = np.copy(Y)

    return PI, U, M_ns, m_ns

# ---------------------EVALUATION---------------------------------

n_sims = 10000
discount = 0.9

# Risk-neutral evaluation
PI, U, M_ns, m_ns = VIA_risk_neutral(S,Q,W,P,R_scaled,eps=2, max_iterations=100)

neutral_practical_reward_sum = 0
neutral_practical_risk = 0
for _ in range(n_sims):
    p_idx = 0
    w_idx = 2
    S_idx = int(np.where(S == S_0)[0][0])
    Q_idx = int(np.where(Q == Q_0)[0][0])
    neutral_practical_risk_count = 0
    for t in range(T):
        fW_t = f(W[w_idx]) * 100
        ehat = E(Q[Q_idx], S[S_idx], fW_t)
        if Q[Q_idx] > ehat:
            neutral_practical_risk_count += 1
        neutral_practical_reward_sum += (discount ** t) * R(Q[Q_idx], P[p_idx], ehat)

        # Transition in endogenous variables
        shat = s_hat(S[S_idx], Q[Q_idx], fW_t)
        q = PI[S_idx, Q_idx, w_idx, p_idx]
        Q_idx = int(np.where(Q == q)[0][0])
        s_ny = find_nearest(S, S[S_idx] + shat)
        S_idx = int(np.where(S == s_ny)[0][0])

        # Transition in exogenous variables
        p_idx = np.random.choice(list(range(len(P))), p=P_trans_price[p_idx, None, :][0])
        w_idx = np.random.choice(list(range(len(W))), p=np.ndarray.flatten(P_trans_wind[w_idx, :, None]))

    neutral_practical_risk += neutral_practical_risk_count/T

RN_risk = neutral_practical_risk/n_sims
RN_reward = neutral_practical_reward_sum/n_sims

print("RN risk", RN_risk)
print("RN reward", RN_reward)

# For a chosen risk-preference level, we now run the algorithm for different values of the risk parameter.
# Next, we simulate practical reward and risk for discrete outcomes, by simulating the process 100 times, and calculating the discounted reward and practical risk.
# We approximate practial reward by a regression (OLS) of the discounted reward on the risk parameter.
# We then choose the best risk parameter value by the program in step 3 in the article


#alphas = np.linspace(0.1, 1, 10)[:-1]
alphas = [0.05,0.01,0.02,0.25,0.5,1,2]

simulated_practical_reward = []
simulated_practical_risk = []


for alpha in alphas:
    PI, U, M_ns, m_ns = VIA_risk_entropic(S,Q,W,P,R_scaled,discount=discount, eps=2, max_iterations=40, risk_param=alpha)

    # Simulating the process
    practical_reward_sum = 0
    practical_risk = 0
    for _ in range(n_sims):
        p_idx = 0
        w_idx = 2
        S_idx = int(np.where(S == S_0)[0][0])
        Q_idx = int(np.where(Q == Q_0)[0][0])
        practical_risk_count = 0
        for t in range(T):
            fW_t = f(W[w_idx])*100
            ehat = E(Q[Q_idx], S[S_idx], fW_t)
            if Q[Q_idx] > ehat:
                practical_risk_count += 1
            practical_reward_sum += (discount ** t) * R(Q[Q_idx], P[p_idx], ehat)

            # Transition in endogenous variables
            shat = s_hat(S[S_idx], Q[Q_idx], fW_t)
            q = PI[S_idx, Q_idx, w_idx, p_idx]
            Q_idx = int(np.where(Q == q)[0][0])
            s_ny = find_nearest(S, S[S_idx] + shat)
            S_idx = int(np.where(S == s_ny)[0][0])

            # Transition in exogenous variables
            p_idx = np.random.choice(list(range(len(P))), p = P_trans_price[p_idx, None, :][0])
            w_idx = np.random.choice(list(range(len(W))), p = np.ndarray.flatten(P_trans_wind[w_idx, :, None]))

        practical_risk += practical_risk_count/T

    simulated_practical_risk.append(practical_risk/n_sims)
    simulated_practical_reward.append(practical_reward_sum/n_sims)

print("risk", simulated_practical_risk)
print("reward", simulated_practical_reward)

#risk_practical = np.polyfit(alphas, simulated_practical_risk, deg = 2)
#reward_practical = np.polyfit(alphas, simulated_practical_reward, deg = 2)

def exp_decay(alpha, a, b, c):
    return a * np.exp(-b * alpha) + c

risk_practical, _ = curve_fit(exp_decay, alphas, simulated_practical_risk, p0=[1.0, 10.0, 0.0])
reward_practical,_ = curve_fit(exp_decay, alphas, simulated_practical_reward, p0=[1.0, 10.0, 0.0])

alpha_range = np.linspace(alphas[0], alphas[-1], 100)
fit_risk = exp_decay(alpha_range, *risk_practical)
fit_reward = exp_decay(alpha_range, *reward_practical)

plt.plot(alphas,simulated_practical_risk, 'o', label='Simulated Data')
plt.plot(alpha_range,fit_risk, label = "fitted risk")
plt.xlabel("Risk-parameter value")
plt.ylabel("Practical risk")
plt.title("Practical risk vs risk parameter value")
plt.legend()
plt.show()

plt.scatter(alphas,simulated_practical_reward, label = "Simulated Data")
plt.plot(alpha_range,fit_reward, label = "fitted reward")
plt.xlabel("Risk-parameter value")
plt.ylabel("Practical reward")
plt.title("Practical reward vs risk parameter value")
plt.legend()
plt.show()

plt.plot(simulated_practical_risk, simulated_practical_reward, 'o-')
for i, param in enumerate(simulated_practical_risk):
    plt.text(simulated_practical_risk[i], simulated_practical_reward[i], f"{param:.2f}", fontsize=9)
plt.xlabel("Practical Risk (P(Q > e))")
plt.ylabel("Expected Reward")
plt.title("Risk-Reward Tradeoff of Policies")
plt.show()


#tradeoffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
tradeoffs = [0.02,0.05,0.08,0.1,0.15,0.2,0.3,0.5,0.8]
optimal_practical_reward = []
optimal_practical_risk = []
optimal_params = []

for tradeoff in tradeoffs:

    def objective(alpha):
        return -exp_decay(alpha[0], *reward_practical)  # We want to maximize reward_practical, so we minimize its negative

    def constraint(alpha):
        return tradeoff - exp_decay(alpha[0], *risk_practical)

    bounds = [(0.001, alphas[-1])]  # Bounds for alpha
    constraints_risk = scipy.optimize.NonlinearConstraint(constraint,0,tradeoff)  # Constraint for risk_practical

    sol = scipy.optimize.minimize(
        objective, x0=[0.5], bounds=[(0.001, alphas[-1])], constraints = [constraints_risk], method='SLSQP'
    )

    #sol = scipy.optimize.linprog(
    #c=-reward_practical,  # Maximize reward_practical (negated for linprog's minimization)
    #A_ub=[risk_practical],  # Linear constraint on risk_practical
    #b_ub=[tradeoff],  # Replace `some_threshold` with the desired upper bound for risk
    #bounds=(0.001, alphas[-1])  # No bounds on the decision variables
    #)

    alpha_hat = sol.x
    optimal_params.append(alpha_hat)

    # Now we can evaluate the risk-neutral policy with the chosen risk parameter
    PI, U, M_ns, m_ns = VIA_risk_entropic(S,Q,W,P,R_scaled,discount=discount, eps=2, max_iterations=40, risk_param = alpha_hat)
    optimal_practical_reward_sum = 0
    optimal_practical_risk_sim = 0
    for _ in range(n_sims):
        p_idx = 0
        w_idx = 2
        S_idx = int(np.where(S == S_0)[0][0])
        Q_idx = int(np.where(Q == Q_0)[0][0])
        optimal_practical_risk_count = 0
        for t in range(T):
            fW_t = f(W[w_idx]) * 100
            ehat = E(Q[Q_idx], S[S_idx], fW_t)
            if Q[Q_idx] > ehat:
                optimal_practical_risk_count += 1
            optimal_practical_reward_sum += (discount ** t) * R(Q[Q_idx], P[p_idx], ehat)

            # Transition in endogenous variables
            shat = s_hat(S[S_idx], Q[Q_idx], fW_t)
            q = PI[S_idx, Q_idx, w_idx, p_idx]
            Q_idx = int(np.where(Q == q)[0][0])
            s_ny = find_nearest(S, S[S_idx] + shat)
            S_idx = int(np.where(S == s_ny)[0][0])

            # Transition in exogenous variables
            p_idx = np.random.choice(list(range(len(P))), p=P_trans_price[p_idx, None, :][0])
            w_idx = np.random.choice(list(range(len(W))), p=np.ndarray.flatten(P_trans_wind[w_idx, :, None]))

        optimal_practical_risk_sim += optimal_practical_risk_count/T

    optimal_practical_risk.append(optimal_practical_risk_sim/n_sims)
    optimal_practical_reward.append(optimal_practical_reward_sum/n_sims)

print("tradeoffs", tradeoffs)
print("Optimal risk param", optimal_params)
print("practical risk", optimal_practical_risk)
print("pracical reward", optimal_practical_reward)
print("practical risk of RN", list(map(lambda x: x/RN_risk, optimal_practical_risk)))
print("practical reward of RN", list(map(lambda x: x/RN_reward, optimal_practical_reward)))
