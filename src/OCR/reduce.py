import queue

import numpy as np

def reduce_domain(RC, pos, idx0, D):
    S = pos.shape
    in_x_r = lambda x: 0 <= x < S[1]
    in_y_r = lambda y: 0 <= y < S[0]
    in_range = lambda idx: in_x_r(idx[1]) and in_y_r(idx[0])

    L = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)]
    L.remove((0, 0))

    D *= 0
    maximal = [idx0, RC[idx0]]
    Q = queue.Queue()
    Q.put(idx0)
    pos[idx0] = 0
    D[idx0] = 1
    k = 0

    while not Q.empty():
        v = Q.get()
        if RC[v] > RC[maximal[0]]:
            maximal[0] = v
            maximal[1] = RC[v]
        for mv in L:
            next_idx = (v[0] + mv[0], v[1] + mv[1])
            if in_range(next_idx) and pos[next_idx] and not D[next_idx]:
                pos[next_idx] = 0
                D[next_idx] = 1
                k += 1
                Q.put(next_idx)

    pos[maximal[0]] = 1

    return maximal[0], k


def reduce(pos, RC, idxs):


    k = 0
    D = np.zeros(shape=pos.shape)
    Res = []
    for idx in idxs:
        # print(idx, k)
        if pos[idx] == 1 and D[idx] == 0:
            remained_idx, tmp_k = reduce_domain(RC, pos, idx, D)
            k += tmp_k
            Res.append(remained_idx)

    return list(set(Res))