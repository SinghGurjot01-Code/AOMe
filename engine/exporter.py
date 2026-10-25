"""Package a saved AI as a .zip with live progress, speed and ETA.

Two formats, both standard Hugging Face layouts that work outside AOMe:
  adapter : the small LoRA adapter + tokenizer. Needs the base model (downloaded by id) to run.
  merged  : the full model with the adapter merged in. Runs anywhere transformers / llama.cpp / vLLM run.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from . import db, library, llm
from .config import AI_DIR, EXPORTS, VERSION

STATE = {}
STAGES = {
    "adapter": [("prepare", "Preparing files", 5), ("package", "Compressing into a zip", 85), ("verify", "Verifying the package", 10)],
    "merged": [("prepare", "Preparing files", 3), ("merge", "Merging your training into the model", 22),
               ("save", "Writing model weights", 25), ("package", "Compressing into a zip", 40), ("verify", "Verifying the package", 10)],
}
STORED = {".safetensors", ".bin", ".gguf", ".zip", ".pt"}


class Cancelled(Exception):
    pass


def view(eid):
    s = STATE.get(eid)
    if not s:
        return None
    el = (s["finished"] or time.time()) - s["started"]
    prog = s["progress"]
    eta = 0
    if s["state"] == "running" and prog > 3:
        eta = el * (100 - prog) / prog
    out = {k: s[k] for k in ("id", "artifact_id", "format", "state", "stage", "stages", "progress", "bytes_done",
                             "bytes_total", "file", "size", "sha256", "error", "name", "folder")}
    out.update(elapsed=round(el, 1), eta=round(eta), speed=s["speed"])
    return out


def _upd(s):
    tw = sum(x["weight"] for x in s["stages"])
    s["progress"] = round(sum(x["weight"] * x["progress"] for x in s["stages"]) / tw, 1)


def _stage(s, key, p=None, state=None):
    for i, x in enumerate(s["stages"]):
        if x["key"] == key:
            if p is not None:
                x["progress"] = p
            if state:
                x["state"] = state
            if state == "active":
                s["stage"] = x["label"]
            if state == "done":
                x["progress"] = 100
    _upd(s)


def start(artifact_id, fmt):
    a = db.one("SELECT * FROM artifacts WHERE id=?", (artifact_id,))
    if not a:
        raise ValueError("That AI was not found.")
    if fmt not in STAGES:
        raise ValueError("Unknown export format.")
    for s in STATE.values():
        if s["state"] == "running":
            raise RuntimeError("Another export is already running.")
    eid = db.new_id()
    m = db.jload(a["manifest"], {})
    nm = library.safe(a["version_name"].replace(" ", "-"), 60)
    s = {"id": eid, "artifact_id": artifact_id, "format": fmt, "state": "running", "stage": "Starting…", "progress": 0.0,
         "bytes_done": 0, "bytes_total": 0, "speed": 0.0, "file": None, "size": 0, "sha256": None, "error": None,
         "name": nm, "folder": str(EXPORTS), "started": time.time(), "finished": None, "cancel": threading.Event(),
         "stages": [{"key": k, "label": l, "weight": w, "progress": 0, "state": "pending"} for k, l, w in STAGES[fmt]]}
    STATE[eid] = s
    threading.Thread(target=_run, args=(s, a, m), daemon=True).start()
    return eid


def cancel(eid):
    s = STATE.get(eid)
    if s and s["state"] == "running":
        s["cancel"].set()
        return True
    return False


def _chk(s):
    if s["cancel"].is_set():
        raise Cancelled()


README = """# {title}

Exported from AOMe (AI Owned by Me).

* Base model: `{base}`
* Method: {method}
* Format: {fmt_desc}

## Use it

{howto}

## Files

{files}

The base model keeps its own license ({lic}). Your training data and this adapter belong to you.
"""

CHAT_ADAPTER = '''"""Chat with the exported adapter. Needs: pip install torch transformers peft"""
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("adapter")
model = AutoPeftModelForCausalLM.from_pretrained("adapter", torch_dtype="auto", device_map="auto")
history = []
while True:
    q = input("You: ").strip()
    if not q:
        break
    history.append({"role": "user", "content": q})
    ids = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(ids, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9)
    a = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print("AI:", a)
    history.append({"role": "assistant", "content": a})
'''

CHAT_MERGED = CHAT_ADAPTER.replace("from peft import AutoPeftModelForCausalLM\n", "").replace(
    'AutoTokenizer.from_pretrained("adapter")', 'AutoTokenizer.from_pretrained("model")').replace(
    "from_pretrained(\"adapter\", torch_dtype", "from_pretrained(\"model\", torch_dtype").replace(
    "model = AutoPeftModelForCausalLM", "from transformers import AutoModelForCausalLM\nmodel = AutoModelForCausalLM").replace(
    "Chat with the exported adapter.", "Chat with the exported model.").replace(" peft", "")


def _run(s, a, m):
    try:
        _do(s, a, m)
        s["state"] = "done"
    except Cancelled:
        s["state"] = "cancelled"
        if s.get("file"):
            Path(s["file"]).unlink(missing_ok=True)
    except Exception as e:
        s["state"] = "error"
        s["error"] = str(e) or e.__class__.__name__
    finally:
        s["finished"] = time.time()
        s["speed"] = 0
        tmp = EXPORTS / f"tmp-{s['id']}"
        shutil.rmtree(tmp, ignore_errors=True)
        llm.free_memory()


def _copy_tree_size(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


def _do(s, a, m):
    fmt = s["format"]
    src = AI_DIR / a["id"] / "adapter"
    if not src.exists():
        raise RuntimeError("The saved adapter files are missing.")
    base_id = a["base_model_id"]
    tmp = EXPORTS / f"tmp-{s['id']}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    root = f"{s['name']}"
    _stage(s, "prepare", 10, "active")
    files = []                                    # (disk path, archive name)
    extra = {}                                    # archive name -> text
    lic = (m.get("base_license") or "see the base model page")
    _chk(s)
    if fmt == "adapter":
        for f in sorted(src.rglob("*")):
            if f.is_file():
                files.append((f, f"{root}/adapter/{f.relative_to(src).as_posix()}"))
        extra[f"{root}/chat.py"] = CHAT_ADAPTER
        howto = ("1. `pip install torch transformers peft`\n2. `python chat.py` (the base model is downloaded from "
                 f"Hugging Face the first time: `{base_id}`).")
        fdesc = "LoRA adapter (small). Needs the base model to run."
        flist = "* `adapter/` the trained adapter and tokenizer\n* `chat.py` a minimal chat script"
    else:
        _stage(s, "prepare", 100, "done")
        _stage(s, "merge", 5, "active")
        from peft import PeftModel
        path = library.find_local(base_id)
        if not path:
            raise RuntimeError(f"The base model {base_id} must be in your Library to export the full model.")
        tok = llm.load_tokenizer(path)
        _stage(s, "merge", 25)
        _chk(s)
        base = llm.load_base(path, dtype=llm.torch().bfloat16, device="cpu")
        _stage(s, "merge", 60)
        _chk(s)
        pm = PeftModel.from_pretrained(base, str(src))
        _stage(s, "merge", 85)
        merged = pm.merge_and_unload()
        _stage(s, "merge", 100, "done")
        _stage(s, "save", 3, "active")
        out = tmp / "model"
        expect = max(1, sum(p.numel() for p in merged.parameters()) * 2)
        done_flag = threading.Event()

        def watch():
            while not done_flag.is_set():
                if out.exists():
                    _stage(s, "save", min(97, _copy_tree_size(out) / expect * 100))
                time.sleep(0.4)
        w = threading.Thread(target=watch, daemon=True)
        w.start()
        try:
            merged.save_pretrained(str(out), safe_serialization=True, max_shard_size="4GB")
            tok.save_pretrained(str(out))
        finally:
            done_flag.set()
        del merged, pm, base
        llm.free_memory()
        _stage(s, "save", 100, "done")
        for f in sorted(out.rglob("*")):
            if f.is_file():
                files.append((f, f"{root}/model/{f.relative_to(out).as_posix()}"))
        extra[f"{root}/chat.py"] = CHAT_MERGED
        howto = ("1. `pip install torch transformers`\n2. `python chat.py`\n\nTo use it with llama.cpp / Ollama convert it to GGUF with "
                 "`python convert_hf_to_gguf.py model/` from the llama.cpp repository.")
        fdesc = "Full model with your training merged in. Works anywhere Hugging Face models work."
        flist = "* `model/` full weights (safetensors), config and tokenizer\n* `chat.py` a minimal chat script"
    info = {"name": a["version_name"], "base_model_id": base_id, "format": fmt, "created_at": a["created_at"],
            "exported_at": time.time(), "method": m.get("method"), "description": m.get("description"),
            "evaluation_metrics": m.get("evaluation_metrics"), "aome_version": VERSION}
    extra[f"{root}/aome-model.json"] = json.dumps(info, indent=2)
    extra[f"{root}/README.md"] = README.format(title=a["version_name"], base=base_id, method=(m.get("method") or "lora").upper(),
                                               fmt_desc=fdesc, howto=howto, files=flist, lic=lic)
    if fmt == "adapter":
        _stage(s, "prepare", 100, "done")
    # ---- package
    total = sum(f.stat().st_size for f, _ in files) + sum(len(v.encode()) for v in extra.values())
    s["bytes_total"], s["bytes_done"] = total, 0
    zname = f"{s['name']}-{'full' if fmt == 'merged' else 'adapter'}.zip"
    zpath = EXPORTS / zname
    if zpath.exists():
        zpath = EXPORTS / f"{zpath.stem}-{s['id'][:4]}.zip"
    s["file"] = str(zpath)
    _stage(s, "package", 1, "active")
    t0, last_t, last_b = time.time(), time.time(), 0
    with zipfile.ZipFile(zpath, "w", allowZip64=True) as zf:
        for path, arc in files:
            zi = zipfile.ZipInfo.from_file(path, arc)
            zi.compress_type = zipfile.ZIP_STORED if path.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            with open(path, "rb") as fin, zf.open(zi, "w", force_zip64=True) as fout:
                while True:
                    _chk(s)
                    buf = fin.read(4 << 20)
                    if not buf:
                        break
                    fout.write(buf)
                    s["bytes_done"] += len(buf)
                    now = time.time()
                    if now - last_t >= 0.4:
                        inst = (s["bytes_done"] - last_b) / (now - last_t)
                        s["speed"] = inst if not s["speed"] else s["speed"] * 0.6 + inst * 0.4
                        last_t, last_b = now, s["bytes_done"]
                    _stage(s, "package", min(99, s["bytes_done"] / max(1, total) * 100))
        for arc, txt in extra.items():
            zf.writestr(arc, txt)
            s["bytes_done"] += len(txt.encode())
    _stage(s, "package", 100, "done")
    # ---- verify
    _stage(s, "verify", 2, "active")
    size = zpath.stat().st_size
    if not zipfile.is_zipfile(zpath):
        raise RuntimeError("The zip failed verification.")
    h, read, last_t, last_b = hashlib.sha256(), 0, time.time(), 0
    s["bytes_total"], s["bytes_done"] = size, 0
    with open(zpath, "rb") as f:
        while True:
            _chk(s)
            buf = f.read(8 << 20)
            if not buf:
                break
            h.update(buf)
            read += len(buf)
            s["bytes_done"] = read
            now = time.time()
            if now - last_t >= 0.4:
                s["speed"] = (read - last_b) / (now - last_t)
                last_t, last_b = now, read
            _stage(s, "verify", min(99, read / max(1, size) * 100))
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
    if f"{root}/README.md" not in names:
        raise RuntimeError("The zip is missing files.")
    s["sha256"], s["size"] = h.hexdigest(), size
    _stage(s, "verify", 100, "done")
    s["stage"] = "Done"
    s["progress"] = 100.0


def reveal(eid):
    s = STATE.get(eid)
    if not s or not s.get("file"):
        return False
    p = Path(s["file"])
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", str(p)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p.parent)])
        return True
    except Exception:
        return False
