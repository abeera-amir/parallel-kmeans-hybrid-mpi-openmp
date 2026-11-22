#!/usr/bin/env python3
"""final_figures.py - final tables + graphs from the cleaned (median) measurements."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# workers -> time (s).  1 worker = the true serial program.
DATA = {
    "Classical (Lloyd), N=4,000,000": {
        "serial": 29.06,   # median of 3 runs
        "OpenMP (1 x p)": {2: 16.14, 4: 11.15},   # median of 3 runs each
        "MPI (p x 1)":    {2: 15.91, 4: 11.47},
        "Hybrid (p/2 x 2)": {4: 11.23},
        "file": "final_lloyd.png"},
    "Optimized (Hamerly), N=4,000,000": {
        "serial": 3.492,   # mean of 3 runs (benchmark.sh)
        "OpenMP (1 x p)": {2: 2.300, 4: 2.148},   # mean of 3 runs each
        "MPI (p x 1)":    {2: 2.313, 4: 2.152},
        "Hybrid (p/2 x 2)": {4: 2.140},
        "file": "final_hamerly.png"},
}
for title, d in DATA.items():
    ser = d["serial"]
    print(f"\n{title}   serial = {ser} s")
    print(f"{'configuration':22}{'workers':>8}{'time(s)':>10}{'speedup':>9}{'efficiency':>11}")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name in ("OpenMP (1 x p)", "MPI (p x 1)", "Hybrid (p/2 x 2)"):
        single = len(d[name]) == 1          # only one measured point (hybrid): plot a marker, no line
        ws = ([] if single else [1]) + sorted(d[name]); ts = ([] if single else [ser]) + [d[name][w] for w in sorted(d[name])]
        sp = [ser / t for t in ts]
        sty = "s" if single else "o-"
        for w, t, s in zip(ws[0 if single else 1:], ts[0 if single else 1:], sp[0 if single else 1:]):
            print(f"{name:22}{w:>8}{t:>10.2f}{s:>9.2f}{s / w:>11.2f}")
        ax[0].plot(ws, ts, sty, ms=9 if single else 6, label=name)
        ax[1].plot(ws, sp, sty, ms=9 if single else 6, label=name)
        ax[2].plot(ws, [s / w for s, w in zip(sp, ws)], sty, ms=9 if single else 6, label=name)
    ax[1].plot([1, 2, 4], [1, 2, 4], "k--", label="ideal")
    for a, t in zip(ax, ("execution time (s)", "speedup", "efficiency")):
        a.set_title(t); a.set_xlabel("workers (MPI processes x OpenMP threads)")
        a.set_xscale("log", base=2); a.set_xticks([1, 2, 4]); a.set_xticklabels([1, 2, 4])
        a.grid(alpha=.3); a.legend()
    fig.suptitle(title + "   (1 worker = serial program)")
    fig.tight_layout(); fig.savefig(d["file"], dpi=140)
print("\nsaved: final_lloyd.png  final_hamerly.png")
