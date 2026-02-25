# scripts/frustration_vs_maxcut.py
from collections import defaultdict
import networkx as nx
from src.alg import atlas_connected_graphs, compare_frustration_vs_maxcut_fast, frustration_index_ALG_budget, maxcut_defect, sorted_edges

# Parameters (match paper)
MAX_N = 7
EDGE_CAP = 12
TIME_BUDGET = 0.1
SAMPLE = None   # None = all eligible graphs
SEED = 1

data, stats = compare_frustration_vs_maxcut_fast(
    max_n=MAX_N,
    sample=SAMPLE if SAMPLE is not None else 10**9,
    seed=SEED,
    max_m_edges_G=EDGE_CAP,
    time_budget_sec=TIME_BUDGET,
)

print(stats)

# Ranges of ell by defect
by_def = defaultdict(set)
for d in data:
    by_def[d["defect"]].add(d["ell"])

multi = sorted([(defect, sorted(list(ells))) for defect, ells in by_def.items() if len(ells) > 1],
               key=lambda t: (-len(t[1]), t[0]))
print("defect values with multiple ℓ values:", len(multi))
print("top few:", multi[:10])

# Find a witness pair with same (n,m,defect) but different ell (best-effort, exact within TIME_BUDGET)
graphs = atlas_connected_graphs(max_n=MAX_N, keep_isolates_out=True)
graphs = [G for G in graphs if (not nx.is_bipartite(G)) and (G.number_of_edges() <= EDGE_CAP)]

bucket = defaultdict(list)
for G in graphs:
    ell = frustration_index_ALG_budget(G, time_budget_sec=TIME_BUDGET)
    if ell is None:
        continue
    defect = maxcut_defect(G)
    key = (G.number_of_nodes(), G.number_of_edges(), defect)
    bucket[key].append((ell, G))

found = False
for key, grp in bucket.items():
    ells = sorted(set(e for e, _ in grp))
    if len(ells) >= 2:
        e1 = ells[0]
        e2 = ells[-1]
        G1 = next(G for e, G in grp if e == e1)
        G2 = next(G for e, G in grp if e == e2)
        print("\nWITNESS key(n,m,defect)=", key, " ell:", e1, "vs", e2)
        print("Edges G:", sorted_edges(G1))
        print("Edges H:", sorted_edges(G2))
        found = True
        break

if not found:
    print("\nNo witness found under current EDGE_CAP / TIME_BUDGET. Increase EDGE_CAP or TIME_BUDGET.")
