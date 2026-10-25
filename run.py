"""Start the AOMe engine and open the app. Everything runs on this PC.

Works three ways:
  python run.py               -> engine + native window (if pywebview is installed) or your browser
  AOMe.exe                    -> same, packaged (no terminal window; errors are shown in a message box + logs/aome.log)
  AOMe.exe --selftest         -> checks that everything is bundled correctly, then exits (add --strict for the GPU/training libs)
"""
import argparse
import json
import multiprocessing
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import webbrowser

FROZEN = bool(getattr(sys, "frozen", False))
ROOT = os.path.dirname(os.path.abspath(__file__))
_LOG = None


# ------------------------------------------------------------------ logging / errors
def log(msg):
    """Write to the log file (and the console when there is one). Never raises."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    try:
        if _LOG:
            _LOG.write(line + "\n")
            _LOG.flush()
    except Exception:
        pass
    try:
        if sys.__stdout__ is not None and sys.__stdout__ is not _LOG:
            print(msg)
    except Exception:
        pass


def log_path():
    return getattr(_LOG, "name", "") or ""


def show_message(title, text, error=True):
    """A real message box on Windows (the exe has no console). Elsewhere: stderr."""
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, str(text), str(title), 0x10 if error else 0x40)
            return
    except Exception:
        pass
    try:
        print(f"{title}: {text}", file=sys.__stderr__ or sys.stderr)
    except Exception:
        pass


def setup_logging():
    """The windowed exe has no console (stdout/stderr are None), which breaks uvicorn, tqdm and print(). Log to a file."""
    global _LOG
    try:
        from engine import config
        logs = config.LOGS
    except Exception:
        logs = None
    if logs is None:
        base = os.environ.get("AOME_HOME") or os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "AOMe")
        logs = os.path.join(base, "logs")
    try:
        os.makedirs(logs, exist_ok=True)
        _LOG = open(os.path.join(str(logs), "aome.log"), "a", encoding="utf-8", buffering=1)
    except Exception:
        _LOG = open(os.path.join(tempfile.gettempdir(), "aome.log"), "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = _LOG
    if sys.stderr is None:
        sys.stderr = _LOG
    import logging
    logging.basicConfig(stream=_LOG, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    def hook(t, v, tb):
        log("UNCAUGHT " + "".join(traceback.format_exception(t, v, tb)))
    sys.excepthook = hook
    threading.excepthook = lambda a: log("THREAD ERROR " + "".join(traceback.format_exception(a.exc_type, a.exc_value, a.exc_traceback)))
    try:
        from engine import config
        ver = config.VERSION
    except Exception:
        ver = "?"
    log(f"--- AOMe {ver} starting (frozen={FROZEN}, python {sys.version.split()[0]}, {sys.platform}) ---")


def fatal(exc):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log("FATAL " + tb)
    last = tb.strip().splitlines()[-1] if tb.strip() else str(exc)
    show_message("AOMe couldn't start",
                 f"{last}\n\nDetails were saved to:\n{log_path() or '(no log file)'}\n\n"
                 "Run  AOMe.exe --selftest  to see which part is missing.")


# ------------------------------------------------------------------ engine helpers
def free_port(preferred):
    for p in [preferred] + list(range(preferred + 1, preferred + 20)):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("No free port found between %d and %d." % (preferred, preferred + 19))


def running_engine(port):
    """True if an AOMe engine already answers on this port (so a second launch just opens the app)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1.5) as r:
            j = json.loads(r.read().decode("utf-8"))
        return bool(j.get("ok")) and "home" in j
    except Exception:
        return False


def start_engine(port):
    """Start uvicorn in a background thread. Returns (server, thread). Raises if it doesn't come up."""
    import uvicorn
    from engine.app import app
    # Explicit loop/http implementations: no dynamic auto-detection, which is fragile in a packaged exe.
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", log_config=None,
                         access_log=False, loop="asyncio", http="h11", ws="none", lifespan="off")
    server = uvicorn.Server(cfg)
    err = {}

    def serve():
        try:
            server.run()
        except BaseException as e:      # noqa: BLE001 - we must see everything
            err["e"] = e
            log("ENGINE CRASHED " + "".join(traceback.format_exception(type(e), e, e.__traceback__)))

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    for _ in range(300):                # up to 30 s
        if server.started:
            return server, t
        if not t.is_alive():
            break
        time.sleep(0.1)
    raise RuntimeError("The AOMe engine did not start" + (f": {err['e']}" if "e" in err else "") + ".")


def open_window(url, wait_for=None):
    """Native window when pywebview works (the .exe build); otherwise the default browser.
    Any failure of the native window falls back to the browser instead of killing the app."""
    webview = None
    try:
        import webview
    except BaseException as e:          # ImportError, or pythonnet / WebView2 failing to load
        log(f"Native window unavailable ({e.__class__.__name__}: {e}); using the browser.")
    if webview is not None:
        try:
            try:
                webview.settings["ALLOW_DOWNLOADS"] = True
            except Exception:
                pass
            from engine import config
            webview.create_window("AOMe", url, width=1440, height=900, min_size=(1100, 700))
            icon = os.path.join(getattr(sys, "_MEIPASS", ROOT), "ui", "assets", "logo-192.png")
            store = str(config.HOME / "webview")
            # private_mode=False + storage_path: keeps your settings (localStorage) between launches.
            for kw in ({"icon": icon, "private_mode": False, "storage_path": store},
                       {"private_mode": False, "storage_path": store}, {}):
                try:
                    webview.start(**kw)
                    return
                except TypeError:
                    continue
        except BaseException as e:
            log("Native window failed, using the browser instead: " + "".join(traceback.format_exception(type(e), e, e.__traceback__)))
    webbrowser.open(url)
    if FROZEN and sys.platform == "win32":
        # No console to Ctrl+C in: keep AOMe alive until the user says so.
        show_message("AOMe", f"AOMe is open in your browser:\n{url}\n\nClick OK to quit AOMe.", error=False)
        return
    if wait_for is not None:
        try:
            while wait_for.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass


# ------------------------------------------------------------------ self test
CORE = ["fastapi", "starlette", "pydantic", "uvicorn", "h11", "psutil", "httpx", "multipart", "sqlite3"]
TRAINING = ["torch", "transformers", "peft", "accelerate", "safetensors", "tokenizers", "huggingface_hub"]
OPTIONAL = ["pynvml", "bitsandbytes", "tkinter", "webview"]


def _tb_tail(e, n=14):
    """The last few traceback lines: shows the exact file/line that failed."""
    out = []
    for chunk in traceback.format_exception(type(e), e, e.__traceback__)[-n:]:
        out.extend(x for x in chunk.rstrip().splitlines() if x.strip())
    return out[-n * 2:]


def selftest(strict):
    """Import everything the app needs and start the engine once. Writes logs/selftest.txt. Exit code 0 = good."""
    import importlib
    lines, bad = [], 0

    def check(group, names, required):
        nonlocal bad
        for n in names:
            try:
                m = importlib.import_module(n)
                lines.append(f"ok    {n} {getattr(m, '__version__', '')}".rstrip())
            except BaseException as e:
                tag = "FAIL " if required else "warn "
                lines.append(f"{tag} {n}: {e.__class__.__name__}: {e}")
                if required or not isinstance(e, ModuleNotFoundError):
                    lines.extend("        " + x for x in _tb_tail(e))
                if required:
                    bad += 1
    lines.append(f"AOMe selftest - python {sys.version.split()[0]} frozen={FROZEN}")
    check("core", CORE, True)
    check("training", TRAINING, strict)
    check("optional", OPTIONAL, False)
    if sys.platform == "win32":
        try:                              # this is exactly what the native window needs: pythonnet + WebView2 DLLs
            import webview.platforms.edgechromium  # noqa: F401
            lines.append("ok    native window stack (pythonnet + WebView2 loader)")
        except BaseException as e:
            lines.append(f"warn  native window stack: {e.__class__.__name__}: {e}  (AOMe would open in the browser)")
            lines.extend("        " + x for x in _tb_tail(e))
    try:
        import transformers  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401 - lazy imports resolve here
        lines.append("ok    transformers.AutoModelForCausalLM / AutoTokenizer")
        import peft
        from peft import LoraConfig, get_peft_model  # noqa: F401
        lines.append("ok    peft.LoraConfig / get_peft_model")
    except BaseException as e:
        lines.append(f"{'FAIL ' if strict else 'warn '} transformers/peft classes: {e.__class__.__name__}: {e}")
        lines.extend("        " + x for x in _tb_tail(e))
        bad += 1 if strict else 0
    try:
        import torch
        lines.append(f"info  torch {torch.__version__}, CUDA build {torch.version.cuda}, GPU usable: {torch.cuda.is_available()}")
        if strict and not torch.version.cuda:
            lines.append("warn  this torch is the CPU-only build (the app will work but training is slow)")
    except BaseException:
        pass
    try:
        port = free_port(18000)
        server, t = start_engine(port)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as r:
            health = json.loads(r.read().decode())
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            page = r.read()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/system/diagnose", timeout=60) as r:
            diag = json.loads(r.read().decode())
        server.should_exit = True
        if b"<html" not in page.lower()[:2000] and b"<!doctype" not in page.lower()[:200]:
            raise RuntimeError("The UI page is missing or empty (ui folder not bundled?).")
        lines.append(f"ok    engine started, health={health.get('ok')}, UI page {len(page)//1024} KB, "
                     f"system check ran ({len(diag.get('items', []))} items)")
    except BaseException as e:
        lines.append(f"FAIL  engine start / UI / API: {e.__class__.__name__}: {e}")
        bad += 1
    lines.append("RESULT: " + ("ALL GOOD" if not bad else f"{bad} PROBLEM(S) - see FAIL lines"))
    text = "\n".join(lines)
    try:
        from engine import config
        (config.LOGS / "selftest.txt").write_text(text, encoding="utf-8")
    except Exception:
        pass
    log(text)                              # also prints when there is a console
    return 0 if not bad else 1


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-window", action="store_true", help="don't open a window/browser")
    ap.add_argument("--selftest", action="store_true", help="check the install/build, then exit")
    ap.add_argument("--strict", action="store_true", help="with --selftest: also require torch/transformers/peft")
    a = ap.parse_args()
    setup_logging()
    if a.selftest:
        return selftest(a.strict)
    if running_engine(a.port):               # already running: don't start a second engine, just open the app
        log(f"AOMe is already running at http://127.0.0.1:{a.port}/ - opening it.")
        if not a.no_window:
            open_window(f"http://127.0.0.1:{a.port}/")
        return 0
    port = free_port(a.port)
    url = f"http://127.0.0.1:{port}/"
    server, t = start_engine(port)
    log(f"AOMe engine running at {url}  (Ctrl+C to quit)")
    if a.no_window:
        try:
            t.join()
        except KeyboardInterrupt:
            pass
        return 0
    open_window(url, wait_for=t)
    server.should_exit = True
    t.join(timeout=3)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()         # REQUIRED in a frozen exe: stops child processes from re-opening the app
    sys.path.insert(0, ROOT)
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException as e:               # noqa: BLE001
        fatal(e)
    for _f in (_LOG, sys.stdout, sys.stderr):
        try:
            if _f:
                _f.flush()
        except Exception:
            pass
    # Hard exit so no background thread (CUDA, downloads) can keep a closed window's process alive.
    os._exit(code or 0)
