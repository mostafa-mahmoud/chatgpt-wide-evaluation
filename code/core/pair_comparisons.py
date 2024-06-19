import random

import numpy as np
import pandas as pd
import scipy.optimize


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def mle(weights, pairs_a, pairs_b, sigma):
    return -np.sum(np.log(sigmoid((weights[pairs_a] - weights[pairs_b]) / sigma)))


def grad(weights, pairs_a, pairs_b, sigma):
    gradients = np.zeros_like(weights)
    M = sigmoid((weights[pairs_b] - weights[pairs_a]) / sigma) / sigma
    for i, j, p_ji in zip(pairs_a, pairs_b, M):
        gradients[i] -= p_ji
        gradients[j] += p_ji
    return gradients


def btl_sigmoid(df, col_a, col_b, msk, sigma=1):
    unique = sorted(list(set(df[col_a].values.tolist() + df[col_b].values.tolist())))
    mp = {x: i for i, x in enumerate(unique)}
    pairs = np.concatenate([df[msk][[col_a, col_b]].values, df[~msk][[col_b, col_a]].values])
    pairs_a = np.array([mp[i] for i, j in pairs])
    pairs_b = np.array([mp[j] for i, j in pairs])
    init = np.zeros(len(unique))

    scores = scipy.optimize.minimize(
        mle,
        init,
        args=(pairs_a, pairs_b, sigma),
        method="Newton-CG",
        jac=grad,
        # tol=10,
        options={'disp': False}
    )
    logits = sigmoid(scores.x / sigma)
    return dict(zip(unique, logits))


def generateSmallWorldGraph(N, M):
    rnd = random.Random(41)
    # Each node is connected to k nearest neighbors, where k is chosen such that N*k is as close to M as possible
    k = M // N
    p = 0.5  # the probability of rewiring each edge
    edges = set()

    # create a regular ring lattice with N nodes and each node connected to k nearest neighbors
    for i in range(N):
        for j in range(1, k // 2 + 1):
            edges.add(frozenset([i, (i + j) % N]))
            edges.add(frozenset([i, (i - j + N) % N]))

    for i in range(N):
        for j in range(1, k // 2 + 1):
            if rnd.random() < p:
                dest = (i + j) % N
                newDest = rnd.randint(0, N - 1)
                if newDest == i or frozenset([i, newDest]) in edges or frozenset([i, newDest]) in edges:
                    continue
                edges.add(frozenset([i, newDest]))

    return [(list(edge)[0], list(edge)[1]) for edge in edges]


def create_pairs_df(df, text_column, score_column):
    N = df.shape[0]
    edges = generateSmallWorldGraph(N, 4*N)
    res = []
    for i, j in edges:
        ri = df.iloc[i]
        rj = df.iloc[j]
        res.append((i, j, ri[text_column], rj[text_column], ri[score_column], rj[score_column]))
    return pd.DataFrame(res, columns=["idx_a", "idx_b", f"{text_column}_a", f"{text_column}_b", f"{score_column}_a", f"{score_column}_b"])
    
