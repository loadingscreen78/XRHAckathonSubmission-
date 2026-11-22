# Running the GPU benchmark on another laptop

> **If this laptop has a coding agent** (Claude Code, Cursor, Codex…): open this folder in it
> and say **"Do the task in AGENT_TASK.md"**. The agent runs everything below, plus the
> feature-parity test, which downloads about 0.5 GB of video and a features archive. The
> manual steps below cover only the benchmark.

Thanks for helping. This takes about 20–40 minutes, mostly the PyTorch download, and
nothing is installed system-wide. Everything goes into a `.venv` folder inside this project folder.

## What you need

- An NVIDIA GPU with a recent driver. Check by running `nvidia-smi` in a terminal: it should
  print a table showing your GPU.
- Python 3.11 from python.org (tick **Add Python to PATH** when installing). 3.12 or 3.13
  also work.
- About 8 GB of free disk on the drive where you put this folder.
- Internet access, for the ~3 GB PyTorch download.

## Steps (Windows)

1. Unzip this folder somewhere with free space, e.g. `D:\camspector`.
2. Open **PowerShell** in that folder: Shift + right-click in the folder, then "Open PowerShell window here".
3. Run:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup_and_bench.ps1
   ```
4. When it prints `DONE. Send this file back: ...results_<PCNAME>.zip`, send that zip file back.

## Steps (Linux / WSL)

```bash
bash setup_and_bench.sh
```
Send back the `results_<hostname>.zip` it prints.

## What it does

1. Prints GPU info (`nvidia-smi`).
2. Creates `.venv`, installs PyTorch 2.14.1 with CUDA 12.6 and the pinned packages.
3. Runs a GPU check: CUDA works, fp16 maths is correct, and a rough speed figure.
4. Generates tiny synthetic test videos and runs the unit tests.
5. Benchmarks 4 video encoders on GPU (fp16) and CPU, using random weights, so nothing
   else is downloaded.
6. Zips the `experiments/` folder.

No personal files are read. Nothing is uploaded anywhere. The results zip contains only
GPU/driver info, package versions, test output and timing numbers.

## If something fails

- **"Python 3.11 not found"**: install it, or pass the path:
  `.\setup_and_bench.ps1 -Python "C:\Path\To\python.exe"`.
- **GPU check says CUDA is not available**: update the NVIDIA driver, then re-run with
  `-SkipInstall` so nothing is reinstalled.
- **Old driver**: the CUDA 12.6 build needs NVIDIA driver 560 or newer. Update the driver.
- Either way, send back the zip. It includes `setup_log.txt` with the error.

To remove everything afterwards, delete the folder.
