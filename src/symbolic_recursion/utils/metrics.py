# utils/metrics.py
"""Graph and semantic metrics utilities."""

from collections import defaultdict, deque
from itertools import combinations
from typing import Dict, Any, List, Set

from symbolic_recursion.core.motif import SymbolicMemoryCore
from symbolic_recursion.embeddings.embedder import embed_text, cosine_sparse


def _strongly_connected_components(graph: Dict[str, List[str]]) -> List[List[str]]:
    idx = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    low: Dict[str, int] = {}
    comps: List[List[str]] = []

    def strong(v: str) -> None:
        nonlocal idx
        indices[v] = low[v] = idx
        idx += 1
        stack.append(v)
        on_stack.add(v)
        for w in graph.get(v, []):
            if w not in indices:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], indices[w])
        if low[v] == indices[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            comps.append(comp)

    for v in graph:
        if v not in indices:
            strong(v)
    return comps


def graph_stats(smc: SymbolicMemoryCore) -> Dict[str, float]:
    """Compute structural metrics of the motif graph without external libs."""
    nodes = set(smc.motifs.keys())
    directed: Dict[str, List[str]] = {n: [] for n in nodes}
    edges: Set[tuple] = set()
    for m in smc.list_motifs():
        for ref in m.references:
            if ref in nodes:
                directed[m.id].append(ref)
                edges.add(tuple(sorted((m.id, ref))))

    V = len(nodes)
    E = len(edges)

    if V <= 1:
        edge_density = 0.0
    else:
        edge_density = (2.0 * E) / (V * (V - 1))

    # Recurrence: nodes part of cycles
    cyc_nodes: Set[str] = set()
    for comp in _strongly_connected_components(directed):
        if len(comp) > 1:
            cyc_nodes.update(comp)
        elif comp and comp[0] in directed.get(comp[0], []):
            cyc_nodes.add(comp[0])
    recurrence_rate = len(cyc_nodes) / V if V else 0.0

    # Undirected adjacency for component/clustering/path metrics
    undirected: Dict[str, Set[str]] = {n: set() for n in nodes}
    for a, b in edges:
        undirected[a].add(b)
        undirected[b].add(a)

    # Connected components
    visited: Set[str] = set()
    comps: List[List[str]] = []
    for n in nodes:
        if n in visited:
            continue
        q = [n]
        visited.add(n)
        comp: List[str] = []
        while q:
            cur = q.pop()
            comp.append(cur)
            for nb in undirected[cur]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        comps.append(comp)
    lcc_size = max((len(c) for c in comps), default=0)
    lcc_fraction = (lcc_size / V) if V else 0.0

    # Clustering coefficient
    coeffs: List[float] = []
    for n, nbrs in undirected.items():
        k = len(nbrs)
        if k < 2:
            continue
        links = 0
        nbr_list = list(nbrs)
        for u, v in combinations(nbr_list, 2):
            if v in undirected[u]:
                links += 1
        coeffs.append(2.0 * links / (k * (k - 1)))
    clustering_coeff = sum(coeffs) / len(coeffs) if coeffs else 0.0

    # Average shortest path within LCC
    if lcc_size > 1:
        largest = max(comps, key=len)
        nodes_list = list(largest)
        total = 0
        pairs = 0
        allowed = set(largest)
        for s in nodes_list:
            dists = {s: 0}
            dq = deque([s])
            while dq:
                v = dq.popleft()
                for nb in undirected[v]:
                    if nb in allowed and nb not in dists:
                        dists[nb] = dists[v] + 1
                        dq.append(nb)
            total += sum(dists.values())
            pairs += len(dists) - 1
        avg_shortest_path = (total / pairs) if pairs else 0.0
    else:
        avg_shortest_path = 0.0

    return {
        "V": float(V),
        "E": float(E),
        "edge_density": float(edge_density),
        "recurrence_rate": float(recurrence_rate),
        "lcc_fraction": float(lcc_fraction),
        "clustering_coeff": float(clustering_coeff),
        "avg_shortest_path": float(avg_shortest_path),
    }


def semantic_stats(smc: SymbolicMemoryCore, prev_centroid: Dict[str, float]) -> Dict[str, Any]:
    """Compute semantic metrics of current motifs.

    Returns a dict with:
    - ``centroid``: current embedding centroid (sparse vector)
    - ``centroid_shift``: 1 - cosine(prev_centroid, centroid)
    - ``cohesion``: mean cosine similarity of motifs to centroid
    """
    vecs = [embed_text(m.content) for m in smc.list_motifs()]
    if not vecs:
        centroid: Dict[str, float] = {}
    else:
        centroid = defaultdict(float)
        for v in vecs:
            for k, val in v.items():
                centroid[k] += val
        n = len(vecs)
        for k in list(centroid.keys()):
            centroid[k] /= n
        centroid = dict(centroid)

    if prev_centroid and centroid:
        centroid_shift = 1.0 - cosine_sparse(prev_centroid, centroid)
    else:
        centroid_shift = 0.0

    if centroid and vecs:
        cohesion = sum(cosine_sparse(v, centroid) for v in vecs) / len(vecs)
    else:
        cohesion = 0.0

    return {
        "centroid": centroid,
        "centroid_shift": centroid_shift,
        "cohesion": cohesion,
    }
