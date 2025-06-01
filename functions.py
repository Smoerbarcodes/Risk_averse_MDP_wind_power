# Risk sensitive value iteration algorithm with entropic risk mapping

def VIA_risk_entropic(S, Q, W, P,R, discount = 0.9,  eps = 1/10, max_iterations=100, risk_param = 0.2):
    len_S, len_Q, len_W, len_P = len(S), len(Q), len(W), len(P)
    U = np.zeros((len_S, len_Q, len_W, len_P))
    Y = np.zeros_like(U)
    PI = np.zeros_like(U)
    M_ns = []
    m_ns = []
    t = 0
    while True:
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
                        Y[s, q, w, p] = max(values)
                        PI[s, q, w, p] = Q[np.argmax(values)]

        if np.max(np.abs(Y-U)) <= eps*(1-discount)/(2*discount):
            break
        if t > max_iterations:
            break
        U = np.copy(Y)

    return PI, U, M_ns, m_ns

# Evaluate the risk-sensitive value function using CVaR which replaces
# V_next in the value iteration with the Conditional Value at Risk (CVaR)
# of the next state values.

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
