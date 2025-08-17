import json, glob

def load_cycles():
    paths = sorted(glob.glob("experiments/logs/novelty_cycle_*.json"))
    return [json.load(open(p)) for p in paths]

def main():
    rows = load_cycles()
    if not rows:
        print("No novelty logs found.")
        return

    print("cycle | V | pursue | mean_novelty | mean_sem | mean_sym | mean_stc | lcc | rec | coh | shift")
    print("------|---|--------|--------------|----------|----------|----------|-----|-----|-----|------")
    for r in rows:
        new = r.get("new_motifs", [])
        nvals = [x["novelty_index"] for x in new] or [0.0]
        sems  = [x["semantic"] for x in new] or [0.0]
        syms  = [x["symbolic"] for x in new] or [0.0]
        stcs  = [x["structural"] for x in new] or [0.0]
        print(f"{r['cycle']:>5} |"
              f"{int(r.get('V',0)):>2} |"
              f"{len(r.get('pursue_queue',[])):>6} |"
              f"{sum(nvals)/len(nvals):>12.3f} |"
              f"{sum(sems)/len(sems):>8.3f} |"
              f"{sum(syms)/len(syms):>8.3f} |"
              f"{sum(stcs)/len(stcs):>8.3f} |"
              f"{r.get('lcc_fraction',0):>3.2f} |"
              f"{r.get('recurrence_rate',0):>3.2f} |"
              f"{r.get('cohesion',0):>3.2f} |"
              f"{r.get('centroid_shift',0):>4.2f}")

    # Top candidates per last cycle
    last = rows[-1]
    cand = last.get("new_motifs", [])
    cand.sort(key=lambda x: x["novelty_index"], reverse=True)
    top = cand[:5]
    print("\nTop novelty candidates (last cycle):")
    for x in top:
        print(f"- {x['id']} | novelty={x['novelty_index']:.3f} (sem={x['semantic']:.3f}, sym={x['symbolic']:.3f}, stc={x['structural']:.3f}) linked={x['linked']}")
