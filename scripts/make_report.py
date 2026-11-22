#!/usr/bin/env python3
"""make_report.py : turns results/results.csv into the tables used in the report
(same layout as Table 3.5 / 3.6 / 3.7 / 5.3 / 5.6 of the course slides) and into PNG graphs."""
import csv, os, sys, collections

rows = list(csv.DictReader(open("results/results.csv")))
for r in rows:
    r["procs"] = int(r["procs"]); r["threads"] = int(r["threads"]); r["time"] = float(r["time"])
    r["n"] = int(r["n"]); r["workers"] = max(1, r["procs"]) * r["threads"]
out = []
def P(s=""):
    print(s); out.append(s)

def table(title, header, body, first_w=24):
    P(); P(title)
    P("".ljust(first_w) + "".join(str(h).rjust(11) for h in header))
    for name, vals in body:
        P(name.ljust(first_w) + "".join((f"{v:.3f}" if isinstance(v, float) else str(v)).rjust(11) for v in vals))

def pick(exp, algo=None, **kw):
    return [r for r in rows if r["exp"] == exp and (algo is None or r["algo"] == algo) and all(r[k] == v for k, v in kw.items())]

# ------------------------------------------------------------------ E1
fig_data = {}
for algo in ("lloyd", "hamerly"):
    E = pick("E1", algo)
    if not E: continue
    ser = [r for r in E if r["procs"] == 0][0]["time"]
    workers = sorted({r["workers"] for r in E if r["procs"] != 0})
    def t(kind, w):
        for r in E:
            if r["procs"] == 0: continue
            if kind == "omp" and r["procs"] == 1 and r["threads"] == w: return r["time"]
            if kind == "mpi" and r["threads"] == 1 and r["procs"] == w and w > 1: return r["time"]
            if kind == "hyb" and r["threads"] == 2 and r["procs"] * 2 == w and w >= 4: return r["time"]
        return None
    kinds = [("pure OpenMP (1 x p)", "omp"), ("pure MPI (p x 1)", "mpi"), ("hybrid (p/2 x 2)", "hyb")]
    table(f"Table A [{algo}]: run times in seconds  (serial = {ser:.3f} s, N={E[0]['n']})", workers,
          [(n, [t(k, w) if t(k, w) is not None else "-" for w in workers]) for n, k in kinds])
    table(f"Table B [{algo}]: speedup S = T_serial / T_p", workers,
          [(n, [ser / t(k, w) if t(k, w) else "-" for w in workers]) for n, k in kinds])
    table(f"Table C [{algo}]: efficiency E = S / p", workers,
          [(n, [ser / t(k, w) / w if t(k, w) else "-" for w in workers]) for n, k in kinds])
    fig_data[algo] = (ser, workers, {n: [(w, ser / t(k, w)) for w in workers if t(k, w)] for n, k in kinds}, E[0]["n"])

# ------------------------------------------------------------------ E2..E5
E = pick("E2")
if E:
    base = [r for r in E if r["accum"] == "private"][0]["time"]
    table(f"Table D: accumulation strategy (threads={E[0]['threads']}, N={E[0]['n']}) - time and slow-down vs private",
          ["time(s)", "x slower"], [(r["accum"], [r["time"], r["time"] / base]) for r in E])
E = pick("E3")
if E:
    table("Table E: loop scheduling / tasking", ["time(s)"], [(r["sched"], [r["time"]]) for r in E])
E = pick("E3chunk")
if E:
    table("Table F: dynamic chunk size (small chunks -> cache lines of lab[] shared between threads)", ["time(s)"], [(r["sched"], [r["time"]]) for r in E])
E = pick("E4")
if E:
    ts = sorted({r["threads"] for r in E})
    table(f"Table G (like Table 5.3): ONE parallel region vs fork-join phases, N={E[0]['n']}, iters={E[0]['iters']}", ts,
          [("one parallel region", [next(r["time"] for r in E if r["region"] == "persistent" and r["threads"] == t) for t in ts]),
           ("fork-join per phase", [next(r["time"] for r in E if r["region"] == "forkjoin" and r["threads"] == t) for t in ts])])
E = pick("E5")
if E:
    ser = [r for r in E if r["procs"] == 0][0]["time"]
    hy = sorted([r for r in E if r["procs"] != 0], key=lambda r: r["threads"])
    table("Table H: choice of K (elbow + silhouette) - serial vs tasks", [r["threads"] for r in hy] , [("hybrid time(s)", [r["time"] for r in hy]), ("speedup", [ser / r["time"] for r in hy])])
    P(f"(serial time = {ser:.3f} s)")
os.makedirs("results", exist_ok=True)
open("results/tables.txt", "w").write("\n".join(out) + "\n")
print("\nwritten: results/tables.txt")

# ------------------------------------------------------------------ graphs
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for algo, (ser, workers, series, n) in fig_data.items():
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for name, pts in series.items():
            ws = [w for w, s in pts]; ss = [s for w, s in pts]
            ax[0].plot(ws, [ser / s for s in ss], "o-", label=name)
            ax[1].plot(ws, ss, "o-", label=name)
            ax[2].plot(ws, [s / w for w, s in pts], "o-", label=name)
        ax[1].plot(workers, workers, "k--", label="ideal")
        ax[0].set_title("execution time (s)"); ax[1].set_title("speedup"); ax[2].set_title("efficiency")
        for a in ax: a.set_xlabel("workers (MPI procs x OpenMP threads)"); a.set_xscale("log", base=2); a.grid(alpha=.3); a.legend()
        fig.suptitle(f"K-means ({algo}) N={n}"); fig.tight_layout(); fig.savefig(f"results/scaling_{algo}.png", dpi=130)
    print("graphs: results/scaling_*.png")
except ImportError:
    print("matplotlib not installed -> no graphs (pip install matplotlib)")
