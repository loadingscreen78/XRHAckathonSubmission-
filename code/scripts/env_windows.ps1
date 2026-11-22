# Dev environment for this laptop. Usage:  . .\code\scripts\env_windows.ps1
# Keeps every cache on D: (C: has ~11 GB free). The conda env sets the same vars on activate.
$env:PIP_CACHE_DIR = "D:\cache\pip"
$env:HF_HOME       = "D:\cache\hf"
$env:TORCH_HOME    = "D:\cache\torch"
$env:TMP           = "D:\cache\tmp"
$env:TEMP          = "D:\cache\tmp"
$env:CONDA_PKGS_DIRS = "D:\cache\conda_pkgs"
$env:PYTHONPATH    = "D:\Samsungprism\code"
$env:PYTHONIOENCODING = "utf-8"
$env:CAMSPECTOR_PYTHON = "D:\miniconda3\envs\camspector\python.exe"
