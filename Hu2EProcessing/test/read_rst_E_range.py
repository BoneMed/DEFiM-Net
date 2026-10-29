"""
Read E (Young's modulus) from RST via rst.materials + rst.mesh.material_type.
Run with: conda run -n pyansys python read_rst_E_range.py
"""

import ansys.mapdl.reader as pymapdl_reader
import numpy as np

rst_path = r"J:\Metastasis_Rob\MV04_L3_VOL_10\MV04_L3_VOL_10\sphere_51\MV04_L3_VOL10_SS25.rst"

print(f"Reading: {rst_path}")
rst = pymapdl_reader.read_binary(rst_path)

print(f"\nResult sets: {rst.nsets}")
print(f"Time values: {rst.time_values}")

# --- Materials ---
materials = rst.materials
print(f"\nNumber of materials: {len(materials)}")

# Show a few sample entries
print("\nSample materials (first 5):")
for mid, props in list(materials.items())[:5]:
    print(f"  Mat {mid}: {props}")

# Extract EX per material ID
mat_ids = rst.mesh.material_type
E_elem = np.array([materials[m]['EX'] for m in mat_ids], dtype=np.float32)

print(f"\n=== E (Young's Modulus) per Element ===")
print(f"Total elements: {len(E_elem)}")
print(f"E range: [{E_elem.min():.4e}, {E_elem.max():.4e}]")
print(f"E mean:  {E_elem.mean():.4e}")
print(f"E median:{np.median(E_elem):.4e}")
print(f"E std:   {E_elem.std():.4e}")

pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
vals = np.percentile(E_elem, pcts)
print("\nE Percentiles:")
for p, v in zip(pcts, vals):
    print(f"  P{p:3d}: {v:.4e}")

# Unique E values
unique_E = np.unique(E_elem)
print(f"\nUnique E values: {len(unique_E)}")
print(f"All unique E (sorted): {unique_E}")

# Mat ID 1000 (sphere) E
if 1000 in materials:
    print(f"\nSphere (Mat 1000) E: {materials[1000].get('EX', 'N/A')}")
print("Done.")
