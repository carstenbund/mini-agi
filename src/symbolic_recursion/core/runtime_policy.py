from __future__ import annotations

class Policy:
    def __init__(
        self,
        link_threshold: float = 0.60,     # cosine sim to auto-link
        dedup_threshold: float = 0.93,    # treat as near-duplicate
        novelty_min: float = 0.10,        # require at-least-this novelty to persist
        max_links_per_motif: int = 3,     # cap fan-out
        symbol_jaccard_cap: float = 0.80  # avoid linking near-clones by tags
    ):
        self.link_threshold = link_threshold
        self.dedup_threshold = dedup_threshold
        self.novelty_min = novelty_min
        self.max_links_per_motif = max_links_per_motif
        self.symbol_jaccard_cap = symbol_jaccard_cap

    def should_persist(self, novelty_index: float) -> bool:
        return novelty_index >= self.novelty_min

    def should_link(self, score: float, jaccard: float, links_added: int) -> bool:
        if links_added >= self.max_links_per_motif:
            return False
        if jaccard >= self.symbol_jaccard_cap:
            return False
        return score >= self.link_threshold

    def is_duplicate(self, top1_sim: float) -> bool:
        return top1_sim >= self.dedup_threshold

