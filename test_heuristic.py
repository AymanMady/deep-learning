import json
import networkx as nx
import numpy as np

# Load graph
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

print("Loaded D and G for 100_121")
print(f"Nodes: {D.number_of_nodes()}, Edges in D: {D.number_of_edges()}, Edges in G: {G.number_of_edges()}")

# Let's try a DFS with randomization to find the longest consistent path
def is_consistent(path, G):
    if not path: return False
    return nx.is_connected(G.subgraph(path))

best_path = []
import time
start = time.time()
# Just generate random paths in D
import random
random.seed(42)

for _ in range(50000):
    node = random.choice(list(D.nodes()))
    path = [node]
    while True:
        succs = list(D.successors(path[-1]))
        if not succs: break
        path.append(random.choice(succs))
    
    # Check all subpaths
    L = len(path)
    for i in range(L):
        for j in range(i + len(best_path), L):
            sub = path[i:j+1]
            if len(sub) > len(best_path) and is_consistent(sub, G):
                best_path = sub

print(f"Random DFS best length: {len(best_path)} in {time.time()-start:.2f}s")
