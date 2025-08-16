#!/bin/bash
set -e

ROOT="symbolic-memory-core"
[ -d "$ROOT" ] || { echo "$ROOT not found. Initialize the repo first."; exit 1; }

mkdir -p "$ROOT/utils" "$ROOT/experiments"
touch "$ROOT/utils/__init__.py" "$ROOT/experiments/__init__.py"

# -------------------------------------------------
# utils/confidence.py — heuristic confidence score
# -------------------------------------------------
cat > "$ROOT/utils/confidence.py" << 'EOF'
from typing import Dict, List
import re
from embeddings.embedder import embed_text, cosine_sparse
from core.motif import SymbolicMemoryCore

HEDGE_RE = re.compile(r"\b(might|maybe|perhaps|possibly|could|unclear|not sure|i think|it seems|appears)\b", re.I)

def _text_quality(text: str) -> float:
    """
    0..1 where higher ~ clearer, less hedging/repetition, some structure.
    Simple, fast, deterministic.
    """
    if not text or not text.strip():
        return 0.0
    t = text.strip()

    # Hedging penalty
    hedges = len(HEDGE_RE.findall(t))
    hedge_pen = min(1.0, hedges / 8.0)  # cap

    # Repetition penalty: repeated 3+ word shingles
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", t)]
    rep_pen = 0.0
    if len(words) >= 9:
        shingles = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        from collections import Counter
        c = Counter(shingles)
        repeats = sum(v-1 for v in c.values() if v > 1)
        rep_pen = min(1.0, repeats / max(1, len(shingles)//5))

    # Structure bonus: presence of bullets / numbering / headings
    structure = 0.0
    if any(line.strip().startswith(("-", "*", "•")) for line in t.splitlines()):
        structure += 0.3
    if re.search(r"^\s*\d+\.", t, re.M):
        structure += 0.3
    if len(t.splitlines()) >= 3:
        structure += 0.2
    structure = min(0.7, structure)

    # Length reasonableness: very short or extremely long is suspicious
    ln = len(words)
    if ln < 30:
        len_pen = 0.4
    elif ln > 1200:
        len_pen = 0.3
    else:
        len_pen = 0.0

    # Base quality
    q = 1.0 - (0.5*hedge_pen + 0.4*rep_pen + 0.3*len_pen)
    q = max(0.0, min(1.0, q + structure))
    return q

def _context_density(smc: SymbolicMemoryCore, text: str, sim_k: int = 5) -> float:
    """
    0..1 indicating how well the text anchors in existing memory.
    Uses max cosine similarity to any motif and count of near neighbors.
    """
    vec = embed_text(text)
    if not smc.motifs or not vec:
        return 0.0
    sims: List[float] = []
    for m in smc.list_motifs():
        sims.append(cosine_sparse(vec, embed_text(m.content)))
    sims.sort(reverse=True)
    max_sim = sims[0] if sims else 0.0
    near = [s for s in sims[:sim_k] if s >= 0.35]
    density = 0.6*max_sim + 0.4*(len(near)/float(sim_k))
    return max(0.0, min(1.0, density))

def _self_rating_fn(prompt_text: str) -> float:
    """
    Placeholder for a model self-rating step (0..1).
    Keep stub-safe by returning a neutral value; you can replace this by
    calling your LLM with a rating prompt and parsing a number.
    """
    return 0.5

def confidence_score(
    smc: SymbolicMemoryCore,
    generated_text: str,
    use_self_rating: bool = False,
    weights = {"text":0.45, "context":0.45, "self":0.10}
) -> Dict[str, float]:
    tq = _text_quality(generated_text)
    cd = _context_density(smc, generated_text)
    sr = _self_rating_fn(generated_text) if use_self_rating else 0.5  # neutral when off
    score = weights["text"]*tq + weights["context"]*cd + weights["self"]*sr
    score = max(0.0, min(1.0, score))
    return {"text_quality": tq, "context_density": cd, "self_rating": sr, "confidence": score}
EOF

# -------------------------------------------------
# Update novelty loop to compute confidence + flag
# -------------------------------------------------
if [ -f "$ROOT/experiments/run_loop_novelty.py" ]; then
  python3 - <<'PY' "$ROOT/experiments/run_loop_novelty.py"
import io,sys,re
p=sys.argv[1]
s=open(p,'r',encoding='utf-8').read()
if 'from utils.confidence import confidence_score' in s:
    print("Already patched:", p); sys.exit(0)
s=s.replace(
    "from utils.metrics import graph_stats, semantic_stats\nfrom utils.novelty import novelty_index\n",
    "from utils.metrics import graph_stats, semantic_stats\nfrom utils.novelty import novelty_index\nfrom utils.confidence import confidence_score\n"
)
s=s.replace(
    '    novelty_threshold = float(cfg.get("novelty_threshold", 0.55))   # pursue threshold\n',
    '    novelty_threshold = float(cfg.get("novelty_threshold", 0.55))   # pursue threshold\n    low_conf_threshold = float(cfg.get("low_confidence_threshold", 0.40))\n'
)
s=s.replace(
    '        for mid in new_ids:\n            m = smc.get_motif(mid)\n            # auto-link by similarity\n            linked_any = False\n',
    '        for mid in new_ids:\n            m = smc.get_motif(mid)\n            # auto-link by similarity\n            linked_any = False\n'
)
# Insert confidence calc + thin_territory flag after novelty
s=re.sub(
r'(n = novelty_index\(smc, m\)\n\s+entry = \{\"id\": mid, \*\*n, \"linked\": linked_any\}\n\s+cycle_log\[\\"new_motifs\\"\]\.append\(entry\)\n\n\s+# pursuit decision\n\s+if n\[\\"novelty_index\\"\] >= novelty_threshold:\n\s+    pursue_queue\.append\(mid\))',
r'conf = confidence_score(smc, m.content)\n            thin = conf["confidence"] < low_conf_threshold\n            entry = {"id": mid, **n, **conf, "linked": linked_any, "thin_territory": thin}\n            cycle_log["new_motifs"].append(entry)\n\n            # pursuit decision\n            if n["novelty_index"] >= novelty_threshold:\n                pursue_queue.append(mid)',
s)
open(p,'w',encoding='utf-8').write(s)
print("✅ Patched", p)
PY
else
  echo "ℹ️ $ROOT/experiments/run_loop_novelty.py not found; skipping patch."
fi

# -------------------------------------------------
# Add a confidence report
# -------------------------------------------------
cat > "$ROOT/experiments/report_confidence.py" << 'EOF'
import json, glob

def load_cycles():
    paths = sorted(glob.glob("experiments/logs/novelty_cycle_*.json"))
    return [json.load(open(p)) for p in paths]

def main():
    rows = load_cycles()
    if not rows:
        print("No novelty logs found.")
        return

    print("cycle | V | mean_conf | mean_txt | mean_ctx | thin_count | thin%")
    print("------|---|-----------|----------|----------|------------|------")
    for r in rows:
        items = r.get("new_motifs", [])
        if not items:
            print(f"{r['cycle']:>5} |{int(r.get('V',0)):>2} |     0.000 |     0.000 |     0.000 |          0 | 0.00")
            continue
        confs = [x.get("confidence",0.0) for x in items]
        txts  = [x.get("text_quality",0.0) for x in items]
        ctxs  = [x.get("context_density",0.0) for x in items]
        thins = sum(1 for x in items if x.get("thin_territory", False))
        thinp = thins/len(items) if items else 0.0
        print(f"{r['cycle']:>5} |{int(r.get('V',0)):>2} | {sum(confs)/len(confs):>9.3f} | {sum(txts)/len(txts):>8.3f} | {sum(ctxs)/len(ctxs):>8.3f} | {thins:>10} | {thinp:>5.2f}")
EOF

echo "✅ Confidence module added and novelty loop patched."
echo
echo "Run the novelty loop, then report confidence:"
echo "  cd $ROOT"
echo "  python3 experiments/run_loop_novelty.py experiments/scenario.json"
echo "  python3 experiments/report_confidence.py"
echo
echo "Adjust thresholds in experiments/scenario.json:"
echo '  { ..., "low_confidence_threshold": 0.40, "novelty_threshold": 0.55 }'
