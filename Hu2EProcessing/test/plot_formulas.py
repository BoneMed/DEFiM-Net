"""
Generate clean formula images for PPT (white background, no color, large text).
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

formulas = {
    "formula_hu_std": r"$HU_{std} = a \cdot HU_{raw} + b$",
    "formula_hu_def": r"$HU = 1000 \times \frac{\mu - \mu_{water}}{\mu_{water} - \mu_{air}}$",
    "formula_rho_qct": r"$\rho_{QCT} = \frac{HU_{std}}{1000}$",
    "formula_rho_ash": r"$\rho_{ash} = 0.8772 \cdot \rho_{QCT} + 0.07895$",
    "formula_E": r"$E = 10200 \cdot \rho_{ash}^{\ 2.01}$",
    "formula_interp": r"$\rho_{tp} = (1 - \alpha) \cdot \rho_{ref} + \alpha \cdot \rho_{target}, \quad \alpha = \frac{tp}{119}$",
}

for name, formula in formulas.items():
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.text(0.5, 0.5, formula, ha='center', va='center',
            fontsize=48, color='black')
    ax.axis('off')
    path = f'H:/pythonproject/PHD_PROGRAM/GCN_Stress/Hu2EProcessing/test/{name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white',
                transparent=True, pad_inches=0.1)
    plt.close()
    print(f"Saved {name}.png")

print("Done.")
