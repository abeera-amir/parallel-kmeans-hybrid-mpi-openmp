# =============================================================================
#  make_graphs.py  -  reads results/timings.csv and results/choose_k.csv
#                     and draws the graphs + writes results/tables.txt
#
#  speedup    S = T_serial / T_parallel   (T_serial = serial time of the SAME algorithm)
#  efficiency E = S / p                   (p = processes x threads)
# =============================================================================
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGOS = ["classic", "kmeanspp", "hamerly"]
ALGO_NAMES = {"classic": "Classic K-Means", "kmeanspp": "K-Means++", "hamerly": "Hamerly"}
SETUPS = ["openmp", "mpi", "hybrid"]
SETUP_NAMES = {"openmp": "Pure OpenMP (1 process x p threads)",
               "mpi": "Pure MPI (p processes x 1 thread)",
               "hybrid": "Hybrid (p/2 processes x 2 threads)"}
COLORS = {"openmp": "#257de0", "mpi": "#f47230", "hybrid": "#00b894"}
MARKERS = {"openmp": "o", "mpi": "s", "hybrid": "D"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.8})

# ---------------- read the timings ----------------
rows = list(csv.DictReader(open("results/timings.csv")))
for r in rows:
    r["workers"] = int(r["workers"])
    r["time"] = float(r["time"])

serial_time = {r["algo"]: r["time"] for r in rows if r["setup"] == "serial"}
for r in rows:
    r["speedup"] = serial_time[r["algo"]] / r["time"]
    r["efficiency"] = r["speedup"] / r["workers"]

def points_of(algo, setup, value):
    # pure MPI and hybrid with p=1 are the same as 1 process x 1 thread
    found = [r for r in rows if r["algo"] == algo and r["setup"] == setup]
    if setup == "mpi":
        found = [r for r in rows if r["algo"] == algo and r["setup"] == "openmp" and r["workers"] == 1] + found
    found.sort(key=lambda r: r["workers"])
    return [r["workers"] for r in found], [r[value] for r in found]

all_workers = sorted(set(r["workers"] for r in rows if r["setup"] != "serial"))

# ---------------- graph 1 and 2: speedup and efficiency ----------------
def line_graph(value, title, ylabel, filename, ideal):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, algo in zip(axes, ALGOS):
        ax.plot(all_workers, [ideal(p) for p in all_workers], "--", color="#999999",
                linewidth=1.5, label="Ideal")
        for setup in SETUPS:
            x, y = points_of(algo, setup, value)
            ax.plot(x, y, marker=MARKERS[setup], markersize=8, linewidth=2,
                    color=COLORS[setup], label=SETUP_NAMES[setup])
        ax.set_title(ALGO_NAMES[algo])
        ax.set_xlabel("Number of workers p (processes x threads)")
        ax.set_xticks(all_workers)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

line_graph("speedup", "Speedup  S = T_serial / T_parallel", "Speedup",
           "results/speedup.png", lambda p: p)
line_graph("efficiency", "Efficiency  E = Speedup / p", "Efficiency",
           "results/efficiency.png", lambda p: 1.0)

# ---------------- graph 3: run time of every setup ----------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, algo in zip(axes, ALGOS):
    mine = [r for r in rows if r["algo"] == algo]
    order = {"serial": 0, "openmp": 1, "mpi": 2, "hybrid": 3}
    mine.sort(key=lambda r: (order[r["setup"]], r["workers"]))
    names, times, colors = [], [], []
    for r in mine:
        if r["setup"] == "serial":
            names.append("Serial"); colors.append("#666666")
        else:
            names.append("%s\n%sx%s" % (r["setup"].replace("openmp", "OpenMP").replace("mpi", "MPI")
                                        .replace("hybrid", "Hybrid"), r["procs"], r["threads"]))
            colors.append(COLORS[r["setup"]])
        times.append(r["time"])
    bars = ax.bar(range(len(times)), times, color=colors, width=0.7)
    for b, t in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, t, "%.2f" % t, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_title(ALGO_NAMES[algo])
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Time (seconds)")
fig.suptitle("Run time (processes x threads)", fontsize=12)
fig.tight_layout()
fig.savefig("results/time.png", dpi=150)
plt.close(fig)

# ---------------- graph 4: the three algorithms compared ----------------
fig, ax = plt.subplots(figsize=(7, 4.2))
best_parallel = {a: min(r["time"] for r in rows if r["algo"] == a and r["setup"] != "serial") for a in ALGOS}
x = range(len(ALGOS))
b1 = ax.bar([i - 0.18 for i in x], [serial_time[a] for a in ALGOS], width=0.34, color="#666666", label="Serial")
b2 = ax.bar([i + 0.18 for i in x], [best_parallel[a] for a in ALGOS], width=0.34, color="#257de0",
            label="Best parallel setup")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), "%.2f" % b.get_height(),
            ha="center", va="bottom", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels([ALGO_NAMES[a] for a in ALGOS])
ax.set_ylabel("Time (seconds)")
ax.set_title("Classic vs K-Means++ vs Hamerly")
ax.grid(axis="x", visible=False)
ax.legend()
fig.tight_layout()
fig.savefig("results/algorithms.png", dpi=150)
plt.close(fig)

# ---------------- graph 5: elbow and silhouette ----------------
k_rows = list(csv.DictReader(open("results/choose_k.csv")))
ks = [int(r["k"]) for r in k_rows]
sse = [float(r["sse"]) for r in k_rows]
sil = [float(r["silhouette"]) for r in k_rows]
chosen, elbow_k, silhouette_k = [int(v) for v in open("results/chosen_k.txt").read().split()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
ax1.plot(ks, sse, marker="o", color="#257de0", linewidth=2)
ax1.axvline(elbow_k, color="#f47230", linestyle="--", linewidth=1.5)
ax1.text(elbow_k + 0.3, max(sse) * 0.9, "elbow: K = %d" % elbow_k, color="#191919")
ax1.set_title("Elbow method")
ax1.set_xlabel("Number of clusters K")
ax1.set_ylabel("SSE (sum of squared distances)")
ax1.set_xticks(ks)
ax2.plot(ks, sil, marker="o", color="#257de0", linewidth=2)
ax2.axvline(silhouette_k, color="#f47230", linestyle="--", linewidth=1.5)
ax2.text(silhouette_k + 0.3, min(sil), "best: K = %d" % silhouette_k, color="#191919")
ax2.set_title("Silhouette method (higher is better)")
ax2.set_xlabel("Number of clusters K")
ax2.set_ylabel("Average silhouette score")
ax2.set_xticks(ks)
fig.suptitle("Choosing K  ->  chosen K = %d" % chosen, fontsize=12)
fig.tight_layout()
fig.savefig("results/choose_k.png", dpi=150)
plt.close(fig)

# ---------------- tables ----------------
with open("results/tables.txt", "w") as out:
    for algo in ALGOS:
        out.write("\n%s   (serial time = %.3f s)\n" % (ALGO_NAMES[algo], serial_time[algo]))
        out.write("%-8s %6s %8s %4s %10s %9s %11s\n" % ("setup", "procs", "threads", "p", "time(s)", "speedup", "efficiency"))
        for r in rows:
            if r["algo"] == algo:
                out.write("%-8s %6s %8s %4d %10.3f %9.2f %11.2f\n" % (r["setup"], r["procs"], r["threads"],
                          r["workers"], r["time"], r["speedup"], r["efficiency"]))
print(open("results/tables.txt").read())
print("Graphs saved: results/choose_k.png, time.png, speedup.png, efficiency.png, algorithms.png")
