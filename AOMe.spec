# PyInstaller spec: builds dist/AOMe/AOMe.exe (folder build; starts faster than one-file).
# Build with build_exe.bat (it installs things in the right order and runs a self-test afterwards).
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

datas, binaries, hidden = [("ui", "ui")], [], []


def grab(pkg):
    """Bundle a package completely (code, data files, DLLs, metadata). Skipped, with a note, if not installed."""
    try:
        d, b, h = collect_all(pkg)
        datas.extend(d); binaries.extend(b); hidden.extend(h)
    except Exception as e:
        print(f"[AOMe.spec] skipped {pkg}: {e}")


for pkg in ("uvicorn", "fastapi", "starlette", "pydantic", "pydantic_core", "anyio", "h11", "httpx", "httpcore",
            "transformers", "peft", "accelerate", "tokenizers", "safetensors", "huggingface_hub", "webview"):
    grab(pkg)
if sys.platform == "win32":          # pywebview's Windows window needs pythonnet + clr_loader (+ WebView2 on the PC)
    for pkg in ("pythonnet", "clr_loader"):
        grab(pkg)
    hidden += ["clr", "webview.platforms.winforms", "webview.platforms.edgechromium"]

# transformers & friends read their dependencies' versions at import time (importlib.metadata);
# without these .dist-info folders they think torch/numpy/... are not installed.
for dist in ("torch", "numpy", "tqdm", "regex", "requests", "packaging", "filelock", "pyyaml", "psutil",
             "nvidia-ml-py", "bitsandbytes", "sympy", "networkx", "jinja2", "fsspec", "typing_extensions",
             "certifi", "idna", "urllib3", "charset_normalizer", "python-multipart", "pywebview"):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

hidden += collect_submodules("engine") + ["pynvml", "psutil", "multipart", "python_multipart", "sqlite3", "httpx", "h11"]
try:                                  # folder picker in Settings uses tkinter (ships with python.org Python on Windows)
    import tkinter  # noqa: F401
    hidden += ["tkinter", "tkinter.filedialog"]
except Exception:
    print("[AOMe.spec] tkinter not available: the 'Browse...' folder picker will be disabled")

a = Analysis(["run.py"], pathex=["."], binaries=binaries, datas=datas, hiddenimports=hidden)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AOMe", console=False, upx=False,
          icon="ui/assets/favicon.ico")
coll = COLLECT(exe, a.binaries, a.datas, name="AOMe", upx=False)
