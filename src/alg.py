# src/alg.py
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import networkx as nx
import numpy as np


# ============================================================
# Core construction: symmetric lift and antisymmetric line graph
# ============================================================

def build_symmetric_lift(G: nx.Graph):
    """
    Build the symmetric lift HL'_2(G) as a graph on directed edges (u,v).
    Returns (HL, directed_edges, index_map).
    """
    directed = []
    for u, v in G.edges():
        directed.append((u, v))
        directed.append((v, u))
    directed = list(dict.fromkeys(directed))  # unique, preserve order

    idx = {e: i for i, e in enumerate(directed)}
    HL = nx.Graph()
    HL.add_nodes_from(directed)

    for (u, v) in directed:
        # shared tail
        for w in G.neighbors(u):
            if w != v:
                HL.add_edge((u, v), (u, w))
        # shared head
        for z in G.neighbors(v):
            if z != u:
                HL.add_edge((u, v), (z, v))

    return HL, directed, idx


def HL_adjacency_numpy(HL: nx.Graph, directed: List[Tuple[int, int]]):
    """networkx -> dense numpy adjacency matrix without scipy."""
    return nx.to_numpy_array(HL, nodelist=directed, dtype=float)


def get_antisymmetric_matrix_via_lift(G: nx.Graph):
    """
    Construct the signed adjacency matrix of A(G) via the lift decomposition:
      M = (1/2) * B^T A(HL'_2(G)) B
    """
    HL, directed, didx = build_symmetric_lift(G)
    A = HL_adjacency_numpy(HL, directed)

    undir = sorted(tuple(sorted(e)) for e in G.edges())
    m = len(undir)

    B = np.zeros((len(directed), m), dtype=float)
    for j, (a, b) in enumerate(undir):
        B[didx[(a, b)], j] = 1.0
        B[didx[(b, a)], j] = -1.0

    M = B.T @ A @ B
    return M / 2.0


def build_antisymmetric_matrix_via_rules(G: nx.Graph):
    """
    Rule-based construction of the signed adjacency matrix of A(G).

    Reference orientation convention:
      For each undirected edge {u,v}, we orient it as u->v where u<v.
    Signs on adjacencies in L(G):
      +1 if two edges meet with same incidence at the shared vertex (both heads or both tails),
      -1 if they meet with opposite incidence (one head, one tail).
    """
    oriented = sorted(tuple(sorted(e)) for e in G.edges())
    n = len(oriented)
    M = np.zeros((n, n), dtype=float)

    for i in range(n):
        u, v = oriented[i]
        for j in range(i + 1, n):
            x, y = oriented[j]
            # must share exactly one endpoint to be adjacent in L(G)
            if len({u, v, x, y}) != 3:
                continue

            # Same-incidence (+1): share smaller endpoint or share larger endpoint
            if (u == x and v != y) or (v == y and u != x):
                M[i, j] = 1.0
            # Opposite-incidence (-1): share smaller-to-larger cross match
            elif (u == y and v != x) or (v == x and u != y):
                M[i, j] = -1.0

    return M + M.T


def ALG_matrix(G: nx.Graph, method: str = "rules"):
    """
    Canonical entry point: return signed adjacency matrix of A(G).
    method: "rules" (default) or "lift".
    """
    if method == "rules":
        return build_antisymmetric_matrix_via_rules(G)
    if method == "lift":
        return get_antisymmetric_matrix_via_lift(G)
    raise ValueError("method must be 'rules' or 'lift'")


# ============================================================
# Spectra / triangle statistics
# ============================================================

def get_spectrum(M, ndigits: int = 10):
    eig = np.linalg.eigvalsh(M)[::-1]
    return [round(float(x), ndigits) for x in eig]


def compare_methods(G: nx.Graph, tol: float = 1e-8):
    M_lift = get_antisymmetric_matrix_via_lift(G)
    M_rules = build_antisymmetric_matrix_via_rules(G)

    spec_lift = np.linalg.eigvalsh(M_lift)
    spec_rules = np.linalg.eigvalsh(M_rules)

    return {
        "matrix_close": bool(np.allclose(M_lift, M_rules, atol=tol)),
        "spectra_close": bool(np.allclose(spec_lift, spec_rules, atol=tol)),
        "max_entry_diff": float(np.max(np.abs(M_lift - M_rules))),
        "max_spec_diff": float(np.max(np.abs(spec_lift - spec_rules))),
    }


def signed_triangle_imbalance(M) -> float:
    """Δ3 = tr(M^3)/6 for signed adjacency M."""
    return float(np.trace(M @ M @ M) / 6.0)


def unbalanced_triangle_stats(M):
    n = M.shape[0]
    pos = neg = 0
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] == 0:
                continue
            for k in range(j + 1, n):
                if M[i, k] == 0 or M[j, k] == 0:
                    continue
                s = M[i, j] * M[j, k] * M[k, i]
                if s > 0:
                    pos += 1
                else:
                    neg += 1
    return {"pos_tri": pos, "neg_tri": neg, "has_unbalanced_tri": (neg > 0)}


def spectrum_summaries(M):
    eig = np.linalg.eigvalsh(M)
    return {
        "lambda_min": float(np.min(eig)),
        "lambda_max": float(np.max(eig)),
        "num_negative": int(np.sum(eig < -1e-9)),
        "gap_to_-2": float(np.min(eig) + 2.0),
    }


# ============================================================
# Graph helpers / keys
# ============================================================

def canon_degseq(G: nx.Graph):
    return tuple(sorted((d for _, d in G.degree()), reverse=True))


def line_graph_spectrum_key(G: nx.Graph, ndigits: int = 10):
    L = nx.line_graph(G)
    A = nx.to_numpy_array(L, dtype=float)
    eig = np.linalg.eigvalsh(A)[::-1]
    return tuple(float(x) for x in np.round(eig, ndigits))


def ALG_spectrum_key(G: nx.Graph, ndigits: int = 10, method: str = "rules"):
    M = ALG_matrix(G, method=method)
    eig = np.linalg.eigvalsh(M)[::-1]
    return tuple(float(x) for x in np.round(eig, ndigits))


def ALG_delta3_val(G: nx.Graph, method: str = "rules"):
    M = ALG_matrix(G, method=method)
    return signed_triangle_imbalance(M)


def atlas_connected_graphs(max_n: int = 7, keep_isolates_out: bool = True):
    """
    Connected graphs from networkx.graph_atlas_g(), relabeled to 0..n-1.
    """
    graphs = []
    for H in nx.graph_atlas_g():
        if H.number_of_nodes() == 0:
            continue
        if H.number_of_nodes() > max_n:
            continue
        G = nx.convert_node_labels_to_integers(H, ordering="sorted")
        if keep_isolates_out and any(d == 0 for _, d in G.degree()):
            continue
        if nx.is_connected(G):
            graphs.append(G)
    return graphs


def G_from_edges(edge_list: List[Tuple[int, int]]):
    G = nx.Graph()
    nodes = set()
    for u, v in edge_list:
        nodes.add(u)
        nodes.add(v)
    G.add_nodes_from(sorted(nodes))
    G.add_edges_from(edge_list)
    return G


def sorted_edges(G: nx.Graph):
    return sorted(tuple(sorted(e)) for e in G.edges())


# ============================================================
# MaxCut defect on G (exact for n<=7)
# ============================================================

def max_cut_size(G: nx.Graph) -> int:
    n = int(G.number_of_nodes())
    nodes = list(G.nodes())
    pos = {nodes[i]: i for i in range(n)}
    best = 0

    for mask in range(1 << (n - 1)):
        cut = 0
        for u, v in G.edges():
            iu = 0 if pos[u] == 0 else (mask >> (pos[u] - 1)) & 1
            iv = 0 if pos[v] == 0 else (mask >> (pos[v] - 1)) & 1
            if iu != iv:
                cut += 1
        if cut > best:
            best = cut
    return int(best)


def maxcut_defect(G: nx.Graph) -> int:
    return int(G.number_of_edges()) - max_cut_size(G)


# ============================================================
# Frustration index ℓ(A(G)) via switching optimization (exact with timeout)
# ============================================================

def frustration_index_from_signed_adj_fast_budget(S, time_budget_sec: float = 0.02) -> Optional[int]:
    """
    Exact frustration index ℓ(Σ) for signed adjacency S (0,±1), with a per-instance timeout.

    Computes:
      ℓ(Σ) = (m - max_{s in {±1}^n} sum_{ij in E} sigma_ij s_i s_j)/2
    where n = |V(Σ)| and m = |E(Σ)|.
    Fixes s_0 = +1 to halve the search.
    Returns None on timeout.
    """
    S = np.array(S, dtype=int)
    n = int(S.shape[0])
    edges = [(i, j, int(S[i, j])) for i in range(n) for j in range(i + 1, n) if S[i, j] != 0]
    m = len(edges)
    if m == 0:
        return 0

    best = -10**18
    start = time.time()

    for mask in range(1 << (n - 1)):
        if (mask & 1023) == 0 and (time.time() - start) > float(time_budget_sec):
            return None

        s = np.ones(n, dtype=int)
        for k in range(1, n):
            if (mask >> (k - 1)) & 1:
                s[k] = -1

        val = 0
        for i, j, sig in edges:
            val += sig * s[i] * s[j]
        if val > best:
            best = val

    return int((m - best) // 2)


def frustration_index_ALG_budget(G: nx.Graph, method: str = "rules", time_budget_sec: float = 0.02) -> Optional[int]:
    S = ALG_matrix(G, method=method)
    return frustration_index_from_signed_adj_fast_budget(S, time_budget_sec=time_budget_sec)


def pearson_float(xs, ys) -> float:
    xs = np.array([float(x) for x in xs], dtype=float)
    ys = np.array([float(y) for y in ys], dtype=float)
    if len(xs) < 2:
        return float("nan")
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    denom = np.sqrt((xs * xs).sum() * (ys * ys).sum())
    return float((xs * ys).sum() / denom) if denom != 0 else float("nan")


# ============================================================
# Reporting helpers used by scripts
# ============================================================

def report_pair(E1, E2, ndigits: int = 10, method: str = "rules") -> Dict[str, Any]:
    G1 = G_from_edges(E1)
    G2 = G_from_edges(E2)
    out = {}
    out["isomorphic"] = nx.is_isomorphic(G1, G2)
    out["n_m"] = (G1.number_of_nodes(), G1.number_of_edges(), G2.number_of_nodes(), G2.number_of_edges())
    out["degseq"] = (canon_degseq(G1), canon_degseq(G2))
    out["SpecL_equal"] = (line_graph_spectrum_key(G1, ndigits) == line_graph_spectrum_key(G2, ndigits))
    out["SpecALG_equal"] = (ALG_spectrum_key(G1, ndigits, method) == ALG_spectrum_key(G2, ndigits, method))
    out["SpecL_1"] = tuple(float(x) for x in line_graph_spectrum_key(G1, ndigits))
    out["SpecL_2"] = tuple(float(x) for x in line_graph_spectrum_key(G2, ndigits))
    out["SpecALG_1"] = tuple(float(x) for x in ALG_spectrum_key(G1, ndigits, method))
    out["SpecALG_2"] = tuple(float(x) for x in ALG_spectrum_key(G2, ndigits, method))
    out["Delta3"] = (ALG_delta3_val(G1, method), ALG_delta3_val(G2, method))
    out["Edges1"] = sorted_edges(G1)
    out["Edges2"] = sorted_edges(G2)
    return out


def compare_frustration_vs_maxcut_fast(
    max_n: int = 7,
    sample: int = 250,
    seed: int = 1,
    max_m_edges_G: int = 12,
    time_budget_sec: float = 0.02,
    method: str = "rules",
):
    """
    Sample from connected atlas graphs with <= max_n vertices, exclude bipartite graphs,
    restrict to |E(G)| <= max_m_edges_G, and compute:
      ell(A(G)) with timeout, and maxcut defect of G exactly.
    """
    graphs = atlas_connected_graphs(max_n=max_n, keep_isolates_out=True)
    graphs = [G for G in graphs if (not nx.is_bipartite(G)) and (int(G.number_of_edges()) <= int(max_m_edges_G))]

    rng = np.random.default_rng(int(seed))
    if sample is not None and int(sample) < len(graphs):
        idxs = rng.choice(len(graphs), size=int(sample), replace=False)
        graphs = [graphs[int(i)] for i in idxs]

    data = []
    timeouts = 0

    for G in graphs:
        ell = frustration_index_ALG_budget(G, method=method, time_budget_sec=float(time_budget_sec))
        if ell is None:
            timeouts += 1
            continue
        defect = maxcut_defect(G)
        data.append({"ell": int(ell), "defect": int(defect), "n": int(G.number_of_nodes()), "m": int(G.number_of_edges())})

    if data:
        mean_ell = float(sum(d["ell"] for d in data)) / float(len(data))
        mean_def = float(sum(d["defect"] for d in data)) / float(len(data))
        corr = pearson_float([d["ell"] for d in data], [d["defect"] for d in data])
    else:
        mean_ell = None
        mean_def = None
        corr = None

    stats = {
        "attempted": int(len(graphs)),
        "completed": int(len(data)),
        "timeouts": int(timeouts),
        "mean_ell": mean_ell,
        "mean_defect": mean_def,
        "pearson_corr": corr,
    }
    return data, stats
