import matplotlib.pyplot as plt

filename = "results-plot.txt"

# -----------------------------
# Parse file dynamically
# -----------------------------
with open(filename, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# Header contains experiment names
header = [h.strip() for h in lines[0].split(",")]
experiments = header[1:]          # everything except "N"

# Storage: {experiment_name: (Ns, values)}
data = {exp: ([], []) for exp in experiments}

# Parse rows
for line in lines[1:]:
    parts = [p.strip() for p in line.split(",")]

    N = float(parts[0])
    values = parts[1:]

    for exp, val in zip(experiments, values):
        data[exp][0].append(N)
        data[exp][1].append(float(val))

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(11.69, 7.27))

for exp in experiments:
    Ns, vals = data[exp]
    plt.plot(Ns, vals, marker="o", label=exp)

plt.xscale("log")
plt.yscale("log")  # remove if you prefer linear y-axis

plt.xlabel("N (problem size)")
plt.ylabel("Bandwidth")
plt.title("Experiment results vs N")

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

plt.grid(True, which="both", linestyle="--")
plt.legend()

plt.tight_layout()
plt.savefig('graph.pgf')
plt.show()