import numpy as np
import matplotlib.pyplot as plt

# Try to set a style, fallback to default if not available
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('ggplot')

data = np.loadtxt("C2H6.dat")
bond_length = data[:, 0]
DFT = data[:, 1]
UMA = data[:, 2]

plt.figure(figsize=(8, 6))

plt.plot(bond_length, DFT, label="wB97X-D/def2-SVP", color="royalblue", linewidth=2, marker='o', markersize=4)
plt.plot(bond_length, UMA, label="UMA", color="darkorange", linestyle="--", linewidth=2, marker='s', markersize=4)

plt.xlabel("C-C Bond Length (Å)", fontsize=12)
plt.ylabel("Energy (kcal/mol)", fontsize=12)
plt.title("C2H6 Dissociation Curve", fontsize=14, fontweight='bold')
plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("C2H6_dissociation_curve.png", dpi=300)
