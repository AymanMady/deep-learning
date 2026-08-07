import json
import networkx as nx
import numpy as np
import random
import time

def load_D(path):
    D = nx.DiGraph()
    edges = []
    with open(path) as f:
        for line in f:
            if line.startswith("N:") or line.startswith("P:") or not line.strip(): continue
            a, b = line.strip().split("-")
            edges.append((int(a), int(b)))
    D.add_edges_from(edges)
    return D
def load_G(path):
    G = nx.Graph()
    edges = []
    with open(path) as f:
        for line in f:
            if line.startswith("N:") or line.startswith("P:") or not line.strip(): continue
            a, b = line.strip().split("-")
            edges.append((int(a), int(b)))
    G.add_edges_from(edges)
    return G

D = load_D('data/raw/100_121/graphD.txt')
G = load_G('data/raw/100_121/graphG.txt')

def longest_DG_consistent_subpath(path, D, G):
    L = len(path)
    if L <= 1:
        return path
    best = (0, 0)
    for i in range(L):
        parent = {path[i]: path[i]}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        n_components = 1
        for j in range(i + 1, L):
            v = path[j]
            parent[v] = v
            n_components += 1
            for u in path[i:j]:
                if G.has_edge(u, v):
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        union(ru, rv)
                        n_components -= 1
            if n_components == 1 and (j - i) > (best[1] - best[0]):
                best = (i, j)
    return path[best[0]:best[1] + 1]


def test_stochastic():
    nodes = list(D.nodes())
    best_raw, best_consistent = [], []
    start_t = time.time()
    
    # Precompute transitions (mock embeddings as uniform for now, just to test speed and connectivity bonus)
    transition_probs = {}
    for u in nodes:
        succs = list(D.successors(u))
        if succs:
            transition_probs[u] = (np.array(succs), np.ones(len(succs))/len(succs))
            
    for _ in range(5000):
        u = random.choice(nodes)
        path = [u]
        visited = {u}
        while True:
            if u not in transition_probs:
                break
            succs, probs = transition_probs[u]
            valid_idx = [i for i, v in enumerate(succs) if v not in visited]
            if not valid_idx:
                break
            valid_succs = succs[valid_idx]
            valid_probs = probs[valid_idx]
            
            # Bonus
            bonus = np.array([10.0 if any(G.has_edge(x, v) for x in path) else 1.0 for v in valid_succs])
            valid_probs = valid_probs * bonus
            valid_probs /= valid_probs.sum()
            
            u = np.random.choice(valid_succs, p=valid_probs)
            path.append(u)
            visited.add(u)
            
        raw_path = path
        consistent_path = longest_DG_consistent_subpath(raw_path, D, G)
        if len(consistent_path) > len(best_consistent):
            best_consistent = consistent_path
            best_raw = raw_path
            
    print(f"Length: {len(best_consistent)}, Raw: {len(best_raw)}, Time: {time.time()-start_t:.2f}s")
    
test_stochastic()
