"""Graph analytics over the motif field.

Communities, god motifs, surprising connections, and the field metrics
sketched in Project-vision.md (recurrence, symbolic depth, narrative
binding, drift).

The analysis approach is ported from graphify (Louvain community
detection with deterministic ordering, degree-ranked god nodes,
cross-community surprise scoring) but implemented natively on the
standard library only — no networkx, no igraph — so the repo keeps
growing on its own.

Two edge kinds feed the graph:

- ``reference``      an explicit motif link (``MotifNode.references``), weight 1.0
- ``shared_symbol``  implicit affinity between motifs sharing symbols,
                     weight = 0.5 * Jaccard(symbol sets), always below an
                     explicit reference so stated structure dominates

Everything here is deterministic: same motif field in, same partition,
ranking, and report out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore

REFERENCE_WEIGHT = 1.0
SYMBOL_WEIGHT_SCALE = 0.5


@dataclass
class Edge:
    a: str
    b: str
    weight: float
    kind: str  # "reference" | "shared_symbol"


@dataclass
class MotifGraph:
    nodes: List[str] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    adj: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def degree(self, node: str) -> int:
        return len(self.adj.get(node, {}))

    def weighted_degree(self, node: str) -> float:
        return sum(self.adj.get(node, {}).values())

    def total_weight(self) -> float:
        return sum(e.weight for e in self.edges)


def _norm_symbols(m: MotifNode) -> frozenset:
    return frozenset(s.strip().lower() for s in m.symbols if s.strip())


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def build_motif_graph(
    smc: SymbolicMemoryCore,
    include_symbol_edges: bool = True,
) -> MotifGraph:
    """Build an undirected weighted graph over the motif field.

    References to unknown motif ids and self-references are skipped;
    duplicate/bidirectional references collapse into one edge. A
    shared-symbol edge is only added where no explicit reference exists.
    """
    motifs = {m.id: m for m in smc.list_motifs()}
    g = MotifGraph(nodes=sorted(motifs))
    g.adj = {n: {} for n in g.nodes}
    seen: set = set()

    def _add(a: str, b: str, weight: float, kind: str) -> None:
        key = (min(a, b), max(a, b))
        if key in seen:
            return
        seen.add(key)
        g.edges.append(Edge(key[0], key[1], weight, kind))
        g.adj[a][b] = weight
        g.adj[b][a] = weight

    for mid in g.nodes:
        for ref in motifs[mid].references:
            if ref != mid and ref in motifs:
                _add(mid, ref, REFERENCE_WEIGHT, "reference")

    if include_symbol_edges:
        syms = {mid: _norm_symbols(motifs[mid]) for mid in g.nodes}
        for i, a in enumerate(g.nodes):
            for b in g.nodes[i + 1:]:
                jac = _jaccard(syms[a], syms[b])
                if jac > 0.0:
                    _add(a, b, SYMBOL_WEIGHT_SCALE * jac, "shared_symbol")
    return g


def detect_communities(g: MotifGraph, resolution: float = 1.0) -> Dict[str, int]:
    """Louvain community detection, deterministic via sorted node order.

    Returns {motif_id: community_index}. Community indices are renumbered
    by (size desc, smallest member id) so output is stable. Isolated
    motifs each form their own community.
    """
    if not g.nodes:
        return {}
    m2 = 2.0 * g.total_weight()
    if m2 == 0.0:
        return {n: i for i, n in enumerate(g.nodes)}

    # Work on an aggregatable copy: node -> {neighbor: weight}
    nodes = list(g.nodes)
    adj = {n: dict(g.adj[n]) for n in nodes}
    # membership of original motifs, via chained partitions
    partition = {n: n for n in nodes}

    while True:
        comm = {n: n for n in nodes}
        k = {n: sum(adj[n].values()) for n in nodes}
        sigma = dict(k)  # community total weighted degree

        improved_any = False
        moved = True
        while moved:
            moved = False
            for n in nodes:
                cn = comm[n]
                sigma[cn] -= k[n]
                # weight from n into each neighboring community
                links: Dict[str, float] = {}
                for nb, w in adj[n].items():
                    if nb == n:
                        continue
                    links[comm[nb]] = links.get(comm[nb], 0.0) + w
                best_c, best_gain = cn, 0.0
                for c in sorted(links):
                    gain = links[c] - resolution * sigma[c] * k[n] / m2
                    if gain > best_gain + 1e-12:
                        best_c, best_gain = c, gain
                sigma[best_c] += k[n]
                if best_c != cn:
                    comm[n] = best_c
                    moved = True
                    improved_any = True

        if not improved_any:
            break

        # aggregate: communities become nodes
        partition = {mid: comm[partition[mid]] for mid in partition}
        new_nodes = sorted(set(comm.values()))
        new_adj: Dict[str, Dict[str, float]] = {c: {} for c in new_nodes}
        for n in nodes:
            for nb, w in adj[n].items():
                a, b = comm[n], comm[nb]
                new_adj[a][b] = new_adj[a].get(b, 0.0) + w
        # each undirected edge was seen from both ends; halve off-diagonal
        for a in new_nodes:
            for b in list(new_adj[a]):
                if a != b:
                    new_adj[a][b] /= 1.0  # symmetric entries already per-endpoint
        nodes, adj = new_nodes, new_adj

    # renumber deterministically: size desc, then smallest member id
    groups: Dict[str, List[str]] = {}
    for mid, c in partition.items():
        groups.setdefault(c, []).append(mid)
    ordered = sorted(groups.values(), key=lambda ms: (-len(ms), min(ms)))
    return {mid: idx for idx, ms in enumerate(ordered) for mid in ms}


def god_motifs(
    g: MotifGraph, smc: SymbolicMemoryCore, top_n: int = 10
) -> List[dict]:
    """Most-connected motifs — the dominant symbolic attractors."""
    motifs = {m.id: m for m in smc.list_motifs()}
    ranked = sorted(
        g.nodes,
        key=lambda n: (-g.weighted_degree(n), -g.degree(n), n),
    )
    out = []
    for n in ranked[:top_n]:
        if g.degree(n) == 0:
            continue
        m = motifs[n]
        out.append({
            "id": n,
            "symbols": list(m.symbols),
            "thread_id": m.thread_id,
            "degree": g.degree(n),
            "weighted_degree": round(g.weighted_degree(n), 3),
        })
    return out


def surprising_connections(
    g: MotifGraph,
    communities: Dict[str, int],
    smc: SymbolicMemoryCore,
    top_n: int = 10,
) -> List[dict]:
    """Cross-community edges ranked by how unexpected they are.

    Scoring (transparent, additive):
      +2.0  explicit reference crossing communities (stated, yet distant)
      +1.0  shared-symbol edge crossing communities
      +1.0  endpoints come from different threads (cross-thread binding)
      +0..1 symbol disjointness (1 - Jaccard): less overlap = more surprise
      +0.5  peripheral-to-hub: a low-degree motif reaching a top attractor
    """
    motifs = {m.id: m for m in smc.list_motifs()}
    if not g.edges:
        return []
    hub_cut = max((g.weighted_degree(n) for n in g.nodes), default=0.0) * 0.75
    scored = []
    for e in g.edges:
        if communities.get(e.a) == communities.get(e.b):
            continue
        ma, mb = motifs[e.a], motifs[e.b]
        reasons = []
        score = 2.0 if e.kind == "reference" else 1.0
        reasons.append(f"cross-community {e.kind}")
        if ma.thread_id != mb.thread_id:
            score += 1.0
            reasons.append(f"cross-thread ({ma.thread_id} × {mb.thread_id})")
        disjoint = 1.0 - _jaccard(_norm_symbols(ma), _norm_symbols(mb))
        if disjoint > 0.0:
            score += disjoint
            reasons.append(f"symbol disjointness {disjoint:.2f}")
        degs = (g.weighted_degree(e.a), g.weighted_degree(e.b))
        if hub_cut > 0 and min(degs) <= 1.0 and max(degs) >= hub_cut:
            score += 0.5
            reasons.append("peripheral motif reaching a hub")
        scored.append({
            "a": e.a, "b": e.b,
            "a_symbols": list(ma.symbols), "b_symbols": list(mb.symbols),
            "kind": e.kind,
            "score": round(score, 3),
            "reasons": reasons,
        })
    scored.sort(key=lambda d: (-d["score"], d["a"], d["b"]))
    return scored[:top_n]


def field_metrics(
    smc: SymbolicMemoryCore, g: MotifGraph, communities: Dict[str, int]
) -> dict:
    """The Project-vision benchmarks, computed from structure alone."""
    motifs = smc.list_motifs()
    n = len(motifs)
    ref_edges = [e for e in g.edges if e.kind == "reference"]
    sym_edges = [e for e in g.edges if e.kind == "shared_symbol"]
    cross_thread = sum(
        1 for e in ref_edges
        if smc.get_motif(e.a).thread_id != smc.get_motif(e.b).thread_id
    )
    # symbol drift: 1 - Jaccard(original symbols, current symbols), for
    # motifs with recorded history — a structural proxy for coherence drift
    drifts = []
    for m in motifs:
        if m.history:
            first = frozenset(s.strip().lower() for s in m.history[0].symbols if s.strip())
            drifts.append(1.0 - _jaccard(first, _norm_symbols(m)) if (first or m.symbols) else 0.0)
    return {
        "motif_count": n,
        "reference_edges": len(ref_edges),
        "shared_symbol_edges": len(sym_edges),
        "community_count": len(set(communities.values())),
        "motif_recurrence_rate": round(2.0 * len(ref_edges) / n, 3) if n else 0.0,
        "symbolic_depth": round(sum(len(m.history) for m in motifs) / n, 3) if n else 0.0,
        "narrative_binding": round(cross_thread / len(ref_edges), 3) if ref_edges else 0.0,
        "symbol_drift": round(sum(drifts) / len(drifts), 3) if drifts else 0.0,
    }


def analyze_field(
    smc: SymbolicMemoryCore,
    include_symbol_edges: bool = True,
    resolution: float = 1.0,
    top_n: int = 10,
) -> dict:
    """One-call orchestrator: graph -> communities -> rankings -> metrics."""
    g = build_motif_graph(smc, include_symbol_edges=include_symbol_edges)
    communities = detect_communities(g, resolution=resolution)
    return {
        "graph": g,
        "communities": communities,
        "god_motifs": god_motifs(g, smc, top_n=top_n),
        "surprises": surprising_connections(g, communities, smc, top_n=top_n),
        "metrics": field_metrics(smc, g, communities),
    }


def _community_label(members: List[MotifNode]) -> str:
    counts: Dict[str, int] = {}
    for m in members:
        for s in _norm_symbols(m):
            counts[s] = counts.get(s, 0) + 1
    top = sorted(counts, key=lambda s: (-counts[s], s))[:3]
    return ", ".join(top) if top else "(unlabeled)"


def render_report(smc: SymbolicMemoryCore, analysis: Optional[dict] = None) -> str:
    """Render the motif field report as markdown."""
    if analysis is None:
        analysis = analyze_field(smc)
    g: MotifGraph = analysis["graph"]
    communities: Dict[str, int] = analysis["communities"]
    metrics = analysis["metrics"]
    lines = [f"# Motif Field Report — {datetime.utcnow().date().isoformat()}", ""]

    if not g.nodes:
        lines.append("The field is empty — no motifs yet.")
        return "\n".join(lines) + "\n"

    lines += [
        "## Field",
        f"- {metrics['motif_count']} motifs · "
        f"{metrics['reference_edges']} references · "
        f"{metrics['shared_symbol_edges']} symbol affinities · "
        f"{metrics['community_count']} communities",
        f"- recurrence rate {metrics['motif_recurrence_rate']} · "
        f"symbolic depth {metrics['symbolic_depth']} · "
        f"narrative binding {metrics['narrative_binding']} · "
        f"symbol drift {metrics['symbol_drift']}",
        "",
    ]

    if analysis["god_motifs"]:
        lines.append("## God motifs (dominant attractors)")
        for i, gm in enumerate(analysis["god_motifs"], 1):
            lines.append(
                f"{i}. `{gm['id']}` [{', '.join(gm['symbols'])}] — "
                f"{gm['degree']} links (weight {gm['weighted_degree']}) — thread {gm['thread_id']}"
            )
        lines.append("")

    groups: Dict[int, List[str]] = {}
    for mid, c in communities.items():
        groups.setdefault(c, []).append(mid)
    lines.append("## Communities")
    for c in sorted(groups):
        members = [smc.get_motif(mid) for mid in sorted(groups[c])]
        lines.append(f"### Community {c} — {_community_label(members)}")
        for m in members:
            lines.append(f"- `{m.id}` [{', '.join(m.symbols)}] (thread {m.thread_id})")
        lines.append("")

    if analysis["surprises"]:
        lines.append("## Surprising connections")
        for s in analysis["surprises"]:
            lines.append(
                f"- `{s['a']}` [{', '.join(s['a_symbols'])}] ↔ "
                f"`{s['b']}` [{', '.join(s['b_symbols'])}] "
                f"(score {s['score']}: {'; '.join(s['reasons'])})"
            )
        lines.append("")

    return "\n".join(lines) + "\n"
