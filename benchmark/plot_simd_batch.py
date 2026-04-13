import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Match the aesthetic of main_exp.py
sns.set_theme(
    style="whitegrid",
    rc={
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 16,
        "text.usetex": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
    },
)

# Experimental Data (Million ops/s)
# 2M Queries
# ARM64: Scalar 235.4ms (8.49M), Batch 35.4ms (56.5M)
# x86: Scalar 61ms (32.8M), Batch 28ms (71.4M)

data = {
    'Platform': ['ARM64 (M1)', 'ARM64 (M1)', 'x86 (AVX2)', 'x86 (AVX2)'],
    'Method': ['Scalar read()', 'Batched SIMD', 'Scalar read()', 'Batched SIMD'],
    'Throughput': [8.49, 56.5, 32.36, 73.53]
}

df = pd.DataFrame(data)

# 1. Throughput Figure
fig, ax = plt.subplots(figsize=(10, 6))

colors = ["#A9A9A9", "#369596"] # Professional Neutral Gray vs Sphinx-4 Teal
plot = sns.barplot(
    data=df, 
    x='Platform', 
    y='Throughput', 
    hue='Method', 
    palette=colors,
    ax=ax,
    edgecolor='black'
)

ax.set_title("Throughput Comparison: Scalar vs. Batched SIMD", fontsize=20, fontweight='bold', pad=20)
ax.set_ylabel("Throughput (Million ops/s)", fontsize=18, fontweight='bold')
ax.set_xlabel("Architecture", fontsize=18, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=16)

# Add value labels on top of bars
for p in plot.patches:
    if p.get_height() > 0:
        ax.annotate(f'{p.get_height():.1f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("benchmark/simd_batch_throughput.svg")
plt.savefig("benchmark/simd_batch_throughput.png", dpi=300)

# 2. Speedup Figure
speedups = {
    'Platform': ['ARM64 (M1)', 'x86 (AVX2)'],
    'Speedup': [56.5 / 8.49, 73.53 / 32.36]
}
df_speedup = pd.DataFrame(speedups)

fig2, ax2 = plt.subplots(figsize=(8, 6))
plot2 = sns.barplot(
    data=df_speedup, 
    x='Platform', 
    y='Speedup', 
    palette=["#369596", "#369596"], # Sphinx-4 teal
    ax=ax2,
    edgecolor='black'
)

ax2.set_title("Relative Speedup: Vertical SIMD Batching", fontsize=20, fontweight='bold', pad=20)
ax2.set_ylabel("Speedup (x Factor)", fontsize=18, fontweight='bold')
ax2.set_xlabel("Architecture", fontsize=18, fontweight='bold')
ax2.set_ylim(0, 8)
ax2.tick_params(axis='both', which='major', labelsize=16)

for p in plot2.patches:
    ax2.annotate(f'{p.get_height():.2f}x', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points',
                fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig("benchmark/simd_batch_speedup.svg")
plt.savefig("benchmark/simd_batch_speedup.png", dpi=300)

print("Graphs successfully generated in benchmark/ directory.")
