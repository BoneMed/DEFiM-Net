@echo off
REM Install causal-conv1d + mamba-ssm in pytorch-gpu env (Windows, CUDA 11.8 build)
REM Source from locally-patched repos (ciso646 include added for MSVC compat).
setlocal

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 ( echo [ERR] vcvars64 failed & exit /b 1 )

call C:\Users\Administrator\anaconda3\Scripts\activate.bat pytorch-gpu
if errorlevel 1 ( echo [ERR] conda activate failed & exit /b 1 )

set DISTUTILS_USE_SDK=1
set CAUSAL_CONV1D_FORCE_BUILD=TRUE
set MAMBA_FORCE_BUILD=TRUE
set TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0
set PIP_DEFAULT_TIMEOUT=300
chcp 65001 >nul

echo ================ env ================
where python
where nvcc
where cl
echo.

echo ================ pip install causal-conv1d (local, patched) ================
pip install --no-build-isolation --no-deps "H:\pythonproject\PHD_PROGRAM\GCN_Stress\thirdparty\causal-conv1d"
set CC1D_RC=%errorlevel%
echo causal-conv1d exit: %CC1D_RC%
echo.

echo ================ pip install mamba-ssm (local, patched) ================
pip install --no-build-isolation --no-deps "H:\pythonproject\PHD_PROGRAM\GCN_Stress\thirdparty\mamba"
set MAMBA_RC=%errorlevel%
echo mamba-ssm exit: %MAMBA_RC%
echo.

echo ================ verify ================
python -c "import causal_conv1d; print('causal_conv1d', causal_conv1d.__version__); from causal_conv1d import causal_conv1d_fn; print('causal_conv1d_fn=', causal_conv1d_fn)"
python -c "import mamba_ssm; from mamba_ssm import Mamba; print('mamba_ssm', mamba_ssm.__version__, 'Mamba=', Mamba)"
python -c "import torch; from mamba_ssm import Mamba; m = Mamba(d_model=64).cuda(); x = torch.randn(2, 32, 64, device='cuda'); y = m(x); print('forward ok, out shape', tuple(y.shape))"

endlocal
