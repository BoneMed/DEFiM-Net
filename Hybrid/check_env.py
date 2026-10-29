#!/usr/bin/env python3
"""Quick environment check for HPC training."""

import sys
print(f"Python: {sys.version}")

errors = []
warnings = []

# Core
try:
    import torch
    print(f"torch:        {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version:   {torch.version.cuda}")
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError as e:
    errors.append(f"torch: {e}")

try:
    import numpy as np
    print(f"numpy:        {np.__version__}")
except ImportError as e:
    errors.append(f"numpy: {e}")

try:
    import scipy
    print(f"scipy:        {scipy.__version__}")
except ImportError as e:
    errors.append(f"scipy: {e}")

try:
    import tqdm
    print(f"tqdm:         {tqdm.__version__}")
except ImportError as e:
    errors.append(f"tqdm: {e}")

try:
    import matplotlib
    print(f"matplotlib:   {matplotlib.__version__}")
except ImportError as e:
    warnings.append(f"matplotlib: {e} (needed for plots)")

# torch_scatter (SparsePhysics)
try:
    import torch_scatter
    print(f"torch_scatter: OK")
except ImportError as e:
    errors.append(f"torch_scatter: {e} (required by SparsePhysicsEncoder)")

# PyG (torch_geometric)
try:
    import torch_geometric
    print(f"torch_geometric: {torch_geometric.__version__}")
except ImportError as e:
    warnings.append(f"torch_geometric: {e} (may be needed)")

# Quick functional test
print("\n--- Functional test ---")
try:
    import torch
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from net import DualEncoderUNet_Sparse
    model = DualEncoderUNet_Sparse()
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Model load:   OK ({n/1e6:.2f}M params)")

    # forward pass
    geo = torch.randn(1, 1, 64, 64, 64)
    phys = torch.randn(1, 4, 64, 64, 64)
    mask = (torch.rand(1, 1, 64, 64, 64) > 0.5).float()
    with torch.no_grad():
        out = model(geo, phys, bone_mask=mask)
    print(f"Forward pass: OK (output {tuple(out.shape)})")
except Exception as e:
    errors.append(f"Model test: {e}")

try:
    from dataUtils import PreCutPatchDataset
    print(f"DataLoader:   OK")
except Exception as e:
    errors.append(f"DataLoader: {e}")

try:
    from toolkit import StrainFailLoss
    loss_fn = StrainFailLoss()
    print(f"Loss:         OK")
except Exception as e:
    errors.append(f"Loss: {e}")

# Summary
print("\n" + "="*50)
if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors:
        print(f"  [ERROR] {e}")
else:
    print("All required packages OK")

if warnings:
    print(f"WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  [WARN]  {w}")

if not errors:
    print("\nEnvironment is ready for training.")
else:
    print("\nFix errors before training.")
