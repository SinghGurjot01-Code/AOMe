# AOMe — AI Owned by Me

Local app. Everything runs on your PC: training, testing, exporting. No cloud.

## Run
```
pip install torch --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU; install this FIRST (use the cpu index for CPU-only)
pip install -r requirements.txt
python run.py
```
Opens the app at http://127.0.0.1:8000. With `pywebview` installed it opens as a native window.

## What's inside
- `ui/index.html` — the app (single file) + `ui/vendor/` (PDF reader, Inter font; works offline)
- `engine/` — local Python engine (FastAPI)
  - `trainer.py` LoRA / QLoRA fine-tuning, real checkpoints, stop-at-checkpoint, held-out evaluation vs the base model
  - `inference.py` chat with base models and your trained AIs (streaming)
  - `exporter.py` .zip export with stage progress, speed, ETA, SHA-256
  - `library.py` resumable model downloads from Hugging Face
  - `hardware.py` live CPU / RAM / GPU / VRAM / disk
- Data lives in `%LOCALAPPDATA%\AOMe` (Windows) or `~/.aome`. Override with `AOME_HOME`.

## Export format
A `.zip` with standard Hugging Face layout, a README and `chat.py`.
- **Adapter only** — small; needs the base model.
- **Full model** — your training merged in; works with transformers, llama.cpp (convert to GGUF), vLLM.

## Next
Installer / `.exe` (PyInstaller + pywebview), bundled Python + torch, extensions ("coming soon" in UI).

## Building the Windows exe
1. Install 64-bit **Python 3.12** (3.10-3.12 work; **3.13 and 3.14 do not**) from python.org. It can sit next to a newer Python; `build_exe.bat` finds it via the `py` launcher.
2. Double-click `build_exe.bat`. It makes a clean environment, installs PyTorch **first**, then everything else, builds with PyInstaller and
   finally runs the built exe's self-test. It prints `Done: dist\AOMe\AOMe.exe (self-test passed)` only when everything works.
3. Ship the whole `dist\AOMe` folder (zip it). Users need the Microsoft Edge WebView2 runtime (already part of Windows 10/11);
   without it AOMe opens in the browser instead.

If the exe won't open: run `AOMe.exe --selftest --strict` from a terminal (or look at `%LOCALAPPDATA%\AOMe\logs\selftest.txt`).
It lists every bundled piece with ok / FAIL. Startup errors also appear in a message box and in `%LOCALAPPDATA%\AOMe\logs\aome.log`.

## Windows exe, terminal windows and logs
- `AOMe.exe` (built with `build_exe.bat`) opens **no terminal window**. The engine runs inside the app and writes
  `aome.log` to the `logs` folder inside the data folder (shown in Settings). Your settings are kept between launches.
- `Start AOMe.bat` / `python run.py` **do open a terminal window**: it shows the engine log and closing it quits AOMe.
- On start, AOMe runs a **System check** (also under Settings > Engine > Run system check): green tick / red cross for
  PyTorch, training libraries, NVIDIA driver, GPU, disk space, internet and more, each with a plain fix.
