"""Model library: downloaded base models, with resumable downloads and live progress."""
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from urllib.parse import quote

import httpx

from . import config

HF = "https://huggingface.co"
_lock = threading.Lock()
DL = {"job": None}


def safe(s, n=80):
    return (re.sub(r'[\\/:*?"<>|]+', "-", str(s)).strip() or "model")[:n]


def parse_params_b(s):
    """'7B' -> 7.0, '360M' -> 0.36"""
    if not s:
        return 0.0
    m = re.search(r"([\d.]+)\s*([TBMK])", str(s).replace(",", ""), re.I)
    if not m:
        return 0.0
    v, u = float(m.group(1)), m.group(2).upper()
    return v * 1000 if u == "T" else v if u == "B" else v / 1000 if u == "M" else v / 1e6


def list_items():
    items = []
    for mf in config.MODELS.glob("*/*/aome-model.json"):
        try:
            j = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if j.get("complete") is False:
            continue
        j["_path"] = f"{mf.parent.parent.name}/{mf.parent.name}"
        j["path"] = str(mf.parent)
        j["used"] = bool(j.get("first_used"))
        j["first_run_min"] = first_run_minutes(j.get("bytes") or sum(f.get("size", 0) for f in j.get("files", [])))
        items.append(j)
    return items


def find_local(model_id):
    """Return the folder of a downloaded model whose HF id matches, else None."""
    want = str(model_id or "").strip().lower()
    for it in list_items():
        if str(it.get("id") or "").lower() == want:
            return Path(it["path"])
    try:
        p = Path(model_id)
        if p.exists() and (p / "config.json").exists():
            return p
    except (OSError, ValueError):
        pass
    return None


# ---- first-run tracking: the first time a model is loaded on this PC it is slow (disk -> RAM/VRAM, caches)
def _meta_path(folder):
    return Path(folder) / "aome-model.json"


def is_used(folder) -> bool:
    try:
        return bool(json.loads(_meta_path(folder).read_text(encoding="utf-8")).get("first_used"))
    except Exception:
        return True            # unknown (e.g. a plain folder): don't show the first-run warning


def mark_used(folder) -> None:
    p = _meta_path(folder)
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
        if not j.get("first_used"):
            j["first_used"] = int(time.time() * 1000)
            p.write_text(json.dumps(j, indent=2), encoding="utf-8")
    except Exception:
        pass


def first_run_info(model_id) -> dict:
    """{'first_run': bool, 'minutes': int} for a downloaded model."""
    want = str(model_id or "").strip().lower()
    for it in list_items():
        if str(it.get("id") or "").lower() == want:
            return {"first_run": not it.get("used"), "minutes": it.get("first_run_min") or 1}
    return {"first_run": False, "minutes": 1}


def first_run_minutes(size_bytes) -> int:
    """Rough first-load time. Loading is disk-bound: ~1 min base + ~1.5 min per GB (HDD/laptop worst case)."""
    gb = (size_bytes or 0) / 1073741824
    return max(1, int(round(1 + 1.5 * gb)))


def delete(path_key):
    parts = path_key.split("/")
    if len(parts) != 2 or ".." in path_key:
        raise ValueError("bad path")
    shutil.rmtree(config.MODELS / safe(parts[0]) / safe(parts[1]), ignore_errors=True)


def hf_info(repo, token=None):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    r = httpx.get(f"{HF}/api/models/{repo}?blobs=true", headers=h, timeout=20, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def pick_files(sib):
    """Choose weights + config + tokenizer; skip duplicate formats."""
    names = {s["rfilename"] for s in sib}

    def has_st(d):
        return any(s["rfilename"].startswith(d) and s["rfilename"].endswith(".safetensors") for s in sib)
    out = []
    for s in sib:
        n = s["rfilename"]
        l = n.lower()
        d = n[: n.rfind("/") + 1] if "/" in n else ""
        if n.startswith(".") or "/." in n:
            continue
        if re.search(r"\.(png|jpe?g|gif|webp|mp4|mp3|wav)$", l):
            continue
        if l.endswith(".md") and "license" not in l:
            continue
        if re.search(r"\.(onnx|gguf|h5|msgpack|ot|tflite|mlmodel|mlpackage|pt|pth|ckpt|npz|zip)$", l) \
                and not l.endswith(".safetensors"):
            if not (re.search(r"\.(pt|pth|ckpt)$", l) and not any(x["rfilename"].endswith(".safetensors") for x in sib)):
                continue
        if re.match(r"^(onnx|openvino|coreml|original|flax|tf|gguf)/", l):
            continue
        if l.endswith(".bin") and has_st(d) and not re.search(r"tokenizer|vocab", l):
            continue
        if l.endswith(".safetensors"):
            fp = n[: -len(".safetensors")] + ".fp16.safetensors"
            if fp in names and ".fp16." not in l:
                continue
        out.append(s)
    return out


class Download:
    def __init__(self, meta, files, sub, token):
        self.meta, self.token = meta, token
        self.sub = safe(sub or meta.get("niche") or "Models")
        self.folder = safe(meta.get("name") or meta["id"])
        self.dir = config.MODELS / self.sub / self.folder
        self.files = [{"name": f["rfilename"], "size": f.get("size") or 0, "done": 0, "ok": False} for f in files]
        self.total = sum(f["size"] for f in self.files)
        self.bytes = 0
        self.state = "running"
        self.err = None
        self.cur = ""
        self.hist = []
        self.speed = 0.0
        self.t0 = time.time()
        self.elapsed = 0.0
        self.seg0 = time.time()
        self._stop = threading.Event()
        self._thread = None

    def view(self):
        left = max(0, self.total - self.bytes)
        el = self.elapsed + (time.time() - self.seg0 if self.state == "running" else 0)
        return {"state": self.state, "id": self.meta["id"], "name": self.meta.get("name"), "niche": self.meta.get("niche"),
                "sub": self.sub, "folder": self.folder, "bytes": self.bytes, "total": self.total,
                "speed": self.speed, "eta": (left / self.speed) if self.speed > 1 else 0, "elapsed": el,
                "files_done": sum(1 for f in self.files if f["ok"]), "files_total": len(self.files),
                "current": self.cur, "error": self.err, "hist": self.hist[-60:]}

    def start(self):
        self.state = "running"
        self.err = None
        self.seg0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self):
        if self.state == "running":
            self.state = "paused"
            self.elapsed += time.time() - self.seg0
            self.speed = 0
            self._stop.set()

    def cancel(self):
        was = self.state
        self.state = "cancelled"
        self._stop.set()
        time.sleep(0.2)
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run(self):
        last_t, last_b = time.time(), self.bytes
        hdr = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            for f in self.files:
                if f["ok"]:
                    continue
                if self._stop.is_set():
                    return
                self.cur = f["name"]
                dest = self.dir / f["name"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                part = dest.with_suffix(dest.suffix + ".part")
                have = part.stat().st_size if part.exists() else 0
                if dest.exists() and f["size"] and dest.stat().st_size == f["size"]:
                    self.bytes += f["size"] - f["done"]
                    f["done"], f["ok"] = f["size"], True
                    continue
                self.bytes -= f["done"]
                f["done"] = have
                self.bytes += have
                h = dict(hdr)
                if have:
                    h["Range"] = f"bytes={have}-"
                url = f"{HF}/{self.meta['id']}/resolve/main/" + "/".join(quote(x) for x in f["name"].split("/"))
                with httpx.stream("GET", url, headers=h, follow_redirects=True, timeout=30) as r:
                    if r.status_code in (401, 403):
                        raise RuntimeError("Access denied. This model is gated: accept its license on Hugging Face and add an access token.")
                    if r.status_code not in (200, 206):
                        raise RuntimeError(f"Server returned HTTP {r.status_code} for {f['name']}")
                    mode = "ab"
                    if have and r.status_code == 200:
                        mode = "wb"
                        self.bytes -= f["done"]
                        f["done"] = 0
                    if not f["size"]:
                        cl = int(r.headers.get("content-length") or 0)
                        if cl:
                            f["size"] = cl + (have if r.status_code == 206 else 0)
                            self.total = sum(x["size"] for x in self.files)
                    with open(part, mode) as out:
                        for chunk in r.iter_bytes(1 << 20):
                            if self._stop.is_set():
                                return
                            out.write(chunk)
                            f["done"] += len(chunk)
                            self.bytes += len(chunk)
                            now = time.time()
                            if now - last_t >= 0.5:
                                inst = (self.bytes - last_b) / (now - last_t)
                                self.hist.append(inst)
                                self.hist = self.hist[-120:]
                                w = self.hist[-6:]
                                self.speed = sum(w) / len(w)
                                last_t, last_b = now, self.bytes
                part.replace(dest)
                f["ok"] = True
            if all(f["ok"] for f in self.files) and not self._stop.is_set():
                m = self.meta
                (self.dir / "aome-model.json").write_text(json.dumps({
                    "id": m["id"], "name": m.get("name"), "org": m.get("org"), "niche": m.get("niche"),
                    "tier": m.get("tier"), "params": m.get("params"), "license": m.get("license"),
                    "source": m.get("source"), "files": [{"name": f["name"], "size": f["size"]} for f in self.files],
                    "bytes": self.bytes, "date": int(time.time() * 1000), "complete": True}, indent=2), encoding="utf-8")
                self.state, self.speed = "done", 0
                self.elapsed += time.time() - self.seg0
        except Exception as e:
            if self.state == "running":
                self.state = "error"
                self.err = str(e) or e.__class__.__name__
                self.speed = 0


def start_download(meta, files, sub, token):
    with _lock:
        j = DL["job"]
        if j and j.state in ("running", "paused"):
            raise RuntimeError("Another download is in progress.")
        j = Download(meta, files, sub, token)
        DL["job"] = j
        j.start()
        return j


def ensure_model(model_id, name=None, on_progress=None, cancel=None, token=None):
    """Return a local folder for model_id, downloading it into the library if needed."""
    from .llm import Cancelled
    p = find_local(model_id)
    if p:
        return p
    try:
        info = hf_info(model_id, token)
    except Exception as e:
        raise RuntimeError(f"Couldn't reach Hugging Face to download {model_id} ({e}). "
                           f"Download it from the Library page while online, or check your connection.")
    files = pick_files(info.get("siblings") or [])
    if not files:
        raise RuntimeError(f"No downloadable weights found for {model_id}.")
    meta = {"id": model_id, "name": name or model_id.split("/")[-1], "niche": "Chat LLM",
            "params": "", "license": (info.get("cardData") or {}).get("license", ""),
            "source": f"{HF}/{model_id}"}
    d = Download(meta, files, "Chat LLM", token)
    t = threading.Thread(target=d._run, daemon=True)
    t.start()
    while t.is_alive():
        if cancel is not None and cancel.is_set():
            d._stop.set()
            d.state = "cancelled"
            t.join(5)
            raise Cancelled()
        if on_progress:
            on_progress(d.view())
        time.sleep(1.0)
    if d.state != "done":
        raise RuntimeError(d.err or "Model download failed.")
    return d.dir


# ---------------------------------------------------------------- changing the model folder
MOVE = {"state": "idle", "bytes": 0, "total": 0, "done": 0, "count": 0, "current": "", "error": None, "dest": None}


def check_new_dir(path_str, create=True):
    """Validate a folder the user typed. Returns (Path, free_gb). Raises ValueError with a plain message.
    create=False only checks (used while typing) and never creates anything on disk."""
    raw = str(path_str or "").strip().strip('"')
    if not raw:
        raise ValueError("Type or choose a folder.")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        raise ValueError("Use a full folder path, for example D:\\AOMe\\models.")
    try:
        p = p.resolve()
    except OSError:
        raise ValueError("That path isn't valid.")
    cur = config.MODELS.resolve()
    if p == cur:
        raise ValueError("That's already the model folder.")
    if cur in p.parents or p in cur.parents:
        raise ValueError("Pick a folder that isn't inside (or around) the current model folder.")
    anc = p
    while not anc.exists() and anc != anc.parent:
        anc = anc.parent
    if not anc.exists():
        raise ValueError("That drive or folder isn't available.")
    if create:
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".aome-write-test"
            probe.write_text("ok")
            probe.unlink()
        except Exception as e:
            raise ValueError(f"AOMe can't write to that folder ({e.__class__.__name__}). Choose another one.")
        anc = p
    elif not os.access(str(anc), os.W_OK):
        raise ValueError("AOMe can't write to that location. Choose another folder.")
    free = shutil.disk_usage(str(anc)).free / 1073741824
    return p, round(free, 1)


def models_bytes():
    return sum((it.get("bytes") or sum(f.get("size", 0) for f in it.get("files", []))) for it in list_items())


def is_busy():
    j = DL["job"]
    return bool(j and j.state in ("running", "paused")) or MOVE["state"] == "running"


def apply_new_dir(new_path, move):
    """Switch the library to new_path. With move=True, existing models are moved there in the background."""
    if is_busy():
        raise RuntimeError("Wait for the current download or move to finish first.")
    old = config.MODELS
    p, free = check_new_dir(new_path)
    items = list_items() if move else []
    need = sum((it.get("bytes") or 0) for it in items) / 1073741824
    if move and need and free < need * 1.05 and p.drive != old.resolve().drive:
        raise ValueError(f"Not enough free space there: the models need {need:.1f} GB and the drive has {free:.1f} GB.")
    config.set_models_dir(p)
    MOVE.update(state="idle", bytes=0, total=0, done=0, count=0, current="", error=None, dest=str(p))
    if move and items:
        MOVE.update(state="running", total=sum((it.get("bytes") or 0) for it in items), count=len(items))
        threading.Thread(target=_move_run, args=(old, p, items), daemon=True).start()
    return p


def _copy_tree(src, dst):
    for f in src.rglob("*"):
        rel = f.relative_to(src)
        t = dst / rel
        if f.is_dir():
            t.mkdir(parents=True, exist_ok=True)
            continue
        t.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "rb") as a, open(t, "wb") as b:
            while True:
                buf = a.read(8 << 20)
                if not buf:
                    break
                b.write(buf)
                MOVE["bytes"] += len(buf)
        shutil.copystat(f, t, follow_symlinks=False)


def _move_run(old, new, items):
    try:
        for it in items:
            src = Path(it["path"])
            rel = src.relative_to(old)
            dst = new / rel
            MOVE["current"] = it.get("name") or src.name
            if dst.exists():
                MOVE["error"] = f"{MOVE['current']} already exists in the new folder, so it was left where it was."
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                src.rename(dst)               # instant when both folders are on the same drive
                MOVE["bytes"] += it.get("bytes") or 0
            except OSError:
                try:
                    _copy_tree(src, dst)
                except Exception:
                    shutil.rmtree(dst, ignore_errors=True)   # never leave a half-copied model that looks complete
                    raise
                shutil.rmtree(src, ignore_errors=True)
            MOVE["done"] += 1
        for d in sorted(old.glob("*"), reverse=True):      # tidy empty type folders
            try:
                d.rmdir()
            except OSError:
                pass
        MOVE["state"] = "done"
    except Exception as e:
        MOVE["state"] = "error"
        MOVE["error"] = str(e) or e.__class__.__name__
