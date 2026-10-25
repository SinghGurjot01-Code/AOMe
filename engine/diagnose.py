"""System check: is everything AOMe needs present on this PC? Each item is ok or fail, with a plain fix."""
import importlib.util
import os
import platform
import shutil
import sys

import psutil

from . import config, hardware, llm

FROZEN = bool(getattr(sys, "frozen", False))
TORCH_URL = "https://download.pytorch.org/whl/cu124"


def _item(id, name, ok, detail="", fix="", required=True):
    return {"id": id, "name": name, "status": "ok" if ok else "fail", "required": required,
            "detail": detail, "fix": "" if ok else fix}


def _writable(p):
    try:
        p.mkdir(parents=True, exist_ok=True)
        t = p / ".aome-write-test"
        t.write_text("ok")
        t.unlink()
        return True
    except Exception:
        return False


def _spec(m):
    return importlib.util.find_spec(m) is not None


def _internet():
    try:
        import httpx
        r = httpx.head("https://huggingface.co", timeout=4, follow_redirects=True)
        return r.status_code < 500
    except Exception:
        return False


def _torch_import():
    """(ok, error text). Importing is what fails on a PC missing the Microsoft C++ runtime."""
    if not llm.torch_ok():
        return False, "not installed"
    try:
        import torch  # noqa: F401
        return True, ""
    except Exception as e:
        return False, f"{e.__class__.__name__}: {e}"


def run(refresh=False):
    g = llm.gpu_diagnosis(refresh)
    items = []
    items.append(_item("engine", "AOMe engine", True, f"Running (Python {platform.python_version()})"))
    items.append(_item("datadir", "Data folder is writable", _writable(config.HOME), str(config.HOME),
                       "AOMe can't write to its data folder. Free up space or set the AOME_HOME environment variable to another folder."))
    mw = _writable(config.MODELS)
    try:
        free = shutil.disk_usage(str(config.MODELS)).free / 1073741824
    except OSError:
        free = 0
    items.append(_item("modelsdir", "Model folder is writable", mw, str(config.MODELS),
                       "Choose another model folder in Settings > Storage."))
    items.append(_item("disk", "Enough free disk space", free >= 5, f"{free:.1f} GB free where models are stored",
                       "Models are 1-15 GB each. Free up space or move the model folder to a bigger drive (Settings > Storage)."))
    ram = psutil.virtual_memory().total / 1073741824
    items.append(_item("ram", "Enough memory (RAM)", ram >= 8, f"{ram:.1f} GB installed",
                       "8 GB or more is recommended. Close other programs, and use small models.", required=False))
    t_ok, t_err = _torch_import()
    dll = any(k in t_err.lower() for k in ("dll", "winerror", "1114", "126"))
    fix_t = ("Install the Microsoft Visual C++ Redistributable (x64) from https://aka.ms/vs/17/release/vc_redist.x64.exe, then restart AOMe."
             if dll else
             ("This build of AOMe is missing PyTorch. Download the full AOMe installer again." if FROZEN
              else f"Install PyTorch: pip install torch --index-url {TORCH_URL}"))
    items.append(_item("torch", "PyTorch (the training engine)", t_ok, "Loads correctly" if t_ok else t_err, fix_t))
    libs = [m for m in ("transformers", "peft", "accelerate", "safetensors", "tokenizers") if not _spec(m)]
    items.append(_item("libs", "Training libraries", not libs,
                       "transformers, peft, accelerate, safetensors, tokenizers" if not libs else "Missing: " + ", ".join(libs),
                       "This build of AOMe is incomplete. Download the full installer again." if FROZEN else "Run: pip install -r requirements.txt"))
    nv = g["driver_found"] or g["nvidia_smi"]
    items.append(_item("driver", "NVIDIA graphics driver", nv, g["gpu_name"] or ("Driver found" if nv else ""),
                       "Install the latest NVIDIA driver from nvidia.com/drivers and restart the PC. AMD and Intel GPUs aren't supported. Without a GPU, training runs on the CPU (much slower).",
                       required=False))
    items.append(_item("gpubuild", "GPU-enabled PyTorch", bool(g["torch_cuda_build"]),
                       f"CUDA {g['torch_cuda_build']}" if g["torch_cuda_build"] else "",
                       "The installed PyTorch can't use a GPU. " + ("Download the GPU edition of AOMe." if FROZEN
                                                                  else f"Reinstall it: pip install torch --index-url {TORCH_URL}"),
                       required=False))
    items.append(_item("cuda", "GPU ready for training", g["ok"], g["gpu_name"] or "",
                       g["detail"] or "The GPU can't be used yet; fix the items above, then press Re-check.", required=False))
    bnb = llm.bnb_available() if t_ok else False
    items.append(_item("bnb", "4-bit training (QLoRA) support", bnb, "bitsandbytes available" if bnb else "",
                       "Needs an NVIDIA GPU and the bitsandbytes package. Without it AOMe uses normal LoRA, which needs more GPU memory.",
                       required=False))
    net = _internet()
    items.append(_item("net", "Internet (for downloading models)", net, "Hugging Face reachable" if net else "",
                       "Needed only to download models. Check your connection, VPN or firewall. Already downloaded models work offline.",
                       required=False))
    fails_req = [i for i in items if i["status"] == "fail" and i["required"]]
    fails_opt = [i for i in items if i["status"] == "fail" and not i["required"]]
    return {"items": items, "can_train": not fails_req, "required_failed": len(fails_req),
            "optional_failed": len(fails_opt), "gpu_ok": g["ok"], "os": f"{platform.system()} {platform.release()}"}
