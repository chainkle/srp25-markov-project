import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import csv
from collections import Counter

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Base directory for outputs
base_dir = os.path.expanduser("~/Documents/my_project_outputs")
graph_dir = os.path.join(base_dir, "graphs")
os.makedirs(graph_dir, exist_ok=True)
print("Saving files to:", base_dir)

def generate_conf_model_with_loops(num_nodes: int, degree: int = 3, seed: int = None) -> nx.MultiGraph:
    if num_nodes % 2 != 0 or num_nodes < 2:
        raise ValueError("Number of nodes must be even and >= 2.")
    degree_sequence = [degree] * num_nodes
    G = nx.configuration_model(degree_sequence, seed=seed)
    return G

def rewire_edges(G: nx.MultiGraph, proportion: float = 1.0, keep_self_loops: bool = True) -> nx.MultiGraph:
    H = G.copy()
    edges = list(H.edges())
    num_to_rewire = int(len(edges) * proportion)
    if num_to_rewire == 0:
        return H

    edges_to_rewire = random.sample(edges, num_to_rewire)
    H.remove_edges_from(edges_to_rewire)

    stubs = []
    for u, v in edges_to_rewire:
        stubs += [u, v]
    random.shuffle(stubs)

    new_edges = []
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if not keep_self_loops and u == v:
            continue
        new_edges.append((u, v))

    H.add_edges_from(new_edges)
    return H

def save_graph_figure(G: nx.Graph, step: int, walk_path: list[int] = None):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)

    nx.draw(G, pos, with_labels=True, node_color='lightgray', node_size=300, edge_color='gray')

    if walk_path:
        nx.draw_networkx_nodes(G, pos, nodelist=walk_path, node_color='orange')
        walk_edges = [(walk_path[i], walk_path[i+1]) for i in range(len(walk_path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=walk_edges, edge_color='red', width=2)

    plt.title(f"Graph Step {step}")
    filepath = os.path.join(graph_dir, f"graph_step_{step}.png")
    plt.savefig(filepath)
    plt.close()

def simulate_random_walk(G: nx.Graph, start_node: int, steps: int = 50) -> list[int]:
    walk = [start_node]
    current = start_node
    for _ in range(steps):
        neighbors = list(G.neighbors(current))
        if neighbors:
            current = random.choice(neighbors)
        walk.append(current)
    return walk

def export_walk_to_csv(walk: list[int], filename: str):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Step', 'Node'])
        for i, node in enumerate(walk):
            writer.writerow([i, node])

def export_graph_stats(G: nx.Graph, step: int, filename: str):
    stats = {
        'step': step,
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'num_self_loops': nx.number_of_selfloops(G),
    }

    write_header = not os.path.exists(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(stats)

# ==== Mixing Time Analysis ====

def get_node_frequencies(walk):
    count = Counter(walk)
    total = len(walk)
    return {node: count[node] / total for node in count}

def get_stationary_distribution(G):
    degrees = dict(G.degree())
    total_deg = sum(degrees.values())
    return {node: deg / total_deg for node, deg in degrees.items()}

def total_variation_distance(p, q):
    all_keys = set(p.keys()).union(q.keys())
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in all_keys)

def empirical_distribution(walks, num_nodes):
    counts = [0] * num_nodes
    for walk in walks:
        counts[walk[-1]] += 1
    total = sum(counts)
    return [c / total for c in counts]

def estimate_mixing_time(G, start_node=0, max_steps=500, threshold=0.05):
    walk = simulate_random_walk(G, start_node=start_node, steps=max_steps)
    pi = get_stationary_distribution(G)
    for t in range(10, max_steps, 10):
        freqs = get_node_frequencies(walk[:t])
        tvd = total_variation_distance(freqs, pi)
        if tvd < threshold:
            return t
    return max_steps

# ==== Main Execution ====

num_nodes = 500
degree = 3
num_steps = 150

G = generate_conf_model_with_loops(num_nodes=num_nodes, degree=degree, seed=SEED)
save_graph_figure(G, step=0)
export_graph_stats(G, step=0, filename=os.path.join(base_dir, "graph_stats.csv"))

current_graph = G

for step in range(1, num_steps + 1):
    if step % 2 == 0:
        current_graph = rewire_edges(current_graph, proportion=0.5, keep_self_loops=True)
        print(f"Step {step}: Rewired edges.")
    else:
        print(f"Step {step}: No rewiring.")

    save_graph_figure(current_graph, step=step)
    export_graph_stats(current_graph, step=step, filename=os.path.join(base_dir, "graph_stats.csv"))

# Simulate walk
walk = simulate_random_walk(current_graph, start_node=0, steps=150)
export_walk_to_csv(walk, filename=os.path.join(base_dir, "random_walk.csv"))
save_graph_figure(current_graph, step=num_steps + 1, walk_path=walk)

# Mixing time tracking
pi = get_stationary_distribution(current_graph)
threshold = 0.05
tvds = []

for t in range(10, len(walk), 10):
    freqs = get_node_frequencies(walk[:t])
    tvd = total_variation_distance(freqs, pi)
    tvds.append(tvd)
    print(f"Step {t}: TVD = {tvd:.4f}")

# Mixing time estimate
mixing_time = None
for step in range(10, 501, 10):
    p_t = empirical_distribution([walk[:step]], num_nodes)
    tvd = total_variation_distance(
        {i: p_t[i] for i in range(len(p_t))},
        {i: pi.get(i, 0) for i in range(num_nodes)}
    )
    if tvd < threshold:
        mixing_time = step
        break

print(f"Estimated mixing time: {mixing_time}")

# Plot TVD trajectory
plt.plot(range(10, len(walk), 10), tvds, marker='o')
plt.xlabel("Steps")
plt.ylabel("Total Variation Distance")
plt.title("TVD vs Steps")
plt.grid(True)
plt.show()

print(f"✅ All outputs saved to '{base_dir}'.")
