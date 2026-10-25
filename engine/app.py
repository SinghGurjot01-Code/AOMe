"""AOMe engine HTTP API. Binds to localhost only; serves the UI from ../ui."""
import asyncio
import json
import re
import shutil
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, datasets, db, diagnose, exporter, hardware, inference, library, llm, trainer
from .config import AI_DIR, JOBS, SUPPORTED_MODELS

app = FastAPI(title="AOMe engine", version=config.VERSION)
_LOCAL = re.compile(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$")


@app.middleware("http")
async def local_only(request: Request, call_next):
    """Refuse requests that don't come from this PC's own pages (blocks DNS-rebinding / drive-by sites)."""
    host = (request.headers.get("host") or "").split(":")[0]
    origin = request.headers.get("origin")
    if host not in ("127.0.0.1", "localhost", "[::1]", "testserver"):
        return JSONResponse({"detail": "Local access only."}, status_code=403)
    if origin and origin != "null" and not _LOCAL.match(origin):
        return JSONResponse({"detail": "Cross-site requests are blocked."}, status_code=403)
    return await call_next(request)


app.add_middleware(CORSMiddleware, allow_origin_regex=r"^(https?://(127\.0\.0\.1|localhost)(:\d+)?|null)$",
                   allow_methods=["*"], allow_headers=["*"])


def err(status, msg, errors=None):
    d = {"message": msg}
    if errors:
        d["errors"] = errors
    raise HTTPException(status, d)


# ------------------------------------------------------------ system
@app.get("/api/health")
def health():
    return {"ok": True, "version": config.VERSION, "home": str(config.HOME), "logs": str(config.LOGS),
            "training": trainer.active_job(), "torch": llm.torch_ok()}


@app.get("/api/hardware")
def hw():
    d = hardware.snapshot()
    d["training_job"] = trainer.active_job()
    return d


@app.get("/api/models")
def models():
    return SUPPORTED_MODELS


@app.get("/api/system/diagnose")
def system_diagnose(refresh: bool = False):
    d = diagnose.run(refresh)
    d["hardware"] = hardware.snapshot()
    return d


@app.get("/api/system/gpu-check")
def gpu_check(refresh: bool = False):
    d = llm.gpu_diagnosis(refresh)
    d["hardware"] = hardware.snapshot()
    return d


# ------------------------------------------------------------ settings: where models are stored
def _storage():
    root = config.MODELS
    try:
        free = round(shutil.disk_usage(str(root)).free / 1073741824, 1)
    except OSError:
        free = 0
    return {"models_dir": str(root), "default_models_dir": str(config.DEFAULT_MODELS),
            "is_default": root.resolve() == config.DEFAULT_MODELS.resolve(), "free_gb": free,
            "models_gb": round(library.models_bytes() / 1073741824, 2), "models_count": len(library.list_items()),
            "unavailable_folder": config.MODELS_FALLBACK, "busy": library.is_busy(), "move": dict(library.MOVE)}


@app.get("/api/settings/storage")
def storage_get():
    return _storage()


class StorageIn(BaseModel):
    path: str
    move: bool = True
    check_only: bool = False


@app.post("/api/settings/storage")
def storage_set(b: StorageIn):
    try:
        if b.check_only:
            p, free = library.check_new_dir(b.path, create=False)
            return {"ok": True, "path": str(p), "free_gb": free}
        library.apply_new_dir(b.path, b.move)
    except ValueError as e:
        err(400, str(e))
    except RuntimeError as e:
        err(409, str(e))
    return _storage()


@app.post("/api/settings/storage/reset")
def storage_reset():
    """Go back to the default model folder."""
    try:
        library.apply_new_dir(str(config.DEFAULT_MODELS), True)
    except ValueError as e:
        err(400, str(e))
    except RuntimeError as e:
        err(409, str(e))
    return _storage()


@app.get("/api/settings/storage/move")
def storage_move():
    return dict(library.MOVE)


@app.post("/api/system/pick-folder")
def pick_folder(b: dict | None = None):
    """Open the operating system's folder chooser on this PC (the engine runs locally)."""
    start = str((b or {}).get("start") or config.MODELS)
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            path = filedialog.askdirectory(initialdir=start if Path(start).exists() else str(Path.home()),
                                           title="Choose where AOMe stores models", mustexist=False)
        finally:
            root.destroy()
        return {"path": path or None}
    except Exception:
        return {"path": None, "unsupported": True}


# ------------------------------------------------------------ datasets
@app.post("/api/datasets/validate")
async def ds_validate(files: list[UploadFile] = File(...), task: str = Form("chat")):
    data = [(f.filename, await f.read()) for f in files]
    return datasets.validate(data)[1]


@app.post("/api/datasets/upload")
async def ds_upload(files: list[UploadFile] = File(...), dataset_name: str = Form("My dataset"),
                    task: str = Form("chat"), rights_confirmed: str = Form("false")):
    if rights_confirmed.lower() != "true":
        err(400, "Confirm you have the rights to use this data.")
    data = [(f.filename, await f.read()) for f in files]
    res, v = datasets.create(data, dataset_name.strip() or "My dataset")
    if res is None:
        err(422, "The data has problems.", v["errors"])
    return res


@app.get("/api/datasets")
def ds_list():
    rows = db.q("SELECT * FROM datasets ORDER BY created_at DESC")
    for r in rows:
        r["manifest"] = db.jload(r["manifest"], {})
    return rows


@app.get("/api/datasets/{did}/preview")
def ds_preview(did: str, n: int = 3):
    return datasets.load_split(did, "train")[: max(1, min(n, 10))]


@app.delete("/api/datasets/{did}")
def ds_delete(did: str):
    datasets.delete(did)
    return {"ok": True}


# ------------------------------------------------------------ jobs
class EstimateIn(BaseModel):
    model_id: str
    dataset_id: str | None = None
    sample_count: int = 100
    config: dict = {}


@app.post("/api/jobs/estimate")
def job_estimate(b: EstimateIn):
    return trainer.estimate(b.model_id, b.sample_count, b.config)


class StartIn(BaseModel):
    project_name: str
    dataset_id: str
    model_id: str
    config: dict = {}


@app.post("/api/jobs/start")
def job_start(b: StartIn):
    try:
        jid = trainer.start(b.project_name, b.dataset_id, b.model_id, b.config)
    except (ValueError, RuntimeError) as e:
        err(409 if isinstance(e, RuntimeError) else 400, str(e))
    return {"job_id": jid}


def _job(jid):
    j = db.one("SELECT * FROM jobs WHERE id=?", (jid,))
    if not j:
        err(404, "Job not found.")
    j["config"] = db.jload(j["config"], {})
    j["metrics"] = db.jload(j["metrics"], None)
    j["progress"] = db.jload(j.get("progress"), {})
    j["server_time"] = time.time()
    return j


@app.get("/api/jobs/active")
def job_active():
    jid = trainer.active_job()
    return _job(jid) if jid else None


@app.get("/api/jobs/{jid}")
def job_get(jid: str):
    return _job(jid)


@app.get("/api/jobs/{jid}/events")
def job_events(jid: str, since_id: int = 0):
    rows = db.q("SELECT id,timestamp,event_type,message,data FROM events WHERE job_id=? AND id>? ORDER BY id LIMIT 500",
                (jid, since_id))
    for r in rows:
        r["data"] = db.jload(r["data"], None)
    return rows


@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str):
    return {"ok": trainer.cancel(jid)}


class StopAtIn(BaseModel):
    step: int | None = None


@app.post("/api/jobs/{jid}/stop-at")
def job_stop_at(jid: str, b: StopAtIn):
    return {"ok": trainer.stop_at(jid, b.step)}


class CkIn(BaseModel):
    step: int


@app.post("/api/jobs/{jid}/select-checkpoint")
def job_select_ck(jid: str, b: CkIn):
    try:
        trainer.select_checkpoint(jid, b.step)
    except (ValueError, RuntimeError) as e:
        err(409, str(e))
    return {"ok": True}


@app.get("/api/jobs/{jid}/evaluation")
def job_eval(jid: str):
    p = JOBS / jid / "evaluation.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------ chat / compare
class ChatIn(BaseModel):
    message: str
    history: list[dict] = []
    system: str = ""
    model_type: str = "custom"          # custom | base
    artifact_id: str | None = None
    job_id: str | None = None
    base_model_id: str | None = None
    temperature: float = 0.7
    max_new_tokens: int = 400


@app.post("/api/chat")
async def chat(b: ChatIn):
    hist = [{"role": m["role"], "content": str(m["content"])} for m in b.history
            if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()]
    msgs = hist + [{"role": "user", "content": b.message}]
    q, stop = inference.chat_start(msgs, b.system, b.model_type, b.artifact_id, b.job_id, b.base_model_id,
                                   b.temperature, min(1024, max(16, b.max_new_tokens)))

    async def gen():
        try:
            while True:
                ev = await asyncio.to_thread(q.get)
                if ev is None:
                    break
                yield json.dumps(ev) + "\n"
        finally:
            stop.set()
    return StreamingResponse(gen(), media_type="application/x-ndjson")


class CompareIn(BaseModel):
    question: str
    base_model_id: str
    job_id: str | None = None
    artifact_id: str | None = None


@app.post("/api/models/compare")
def compare(b: CompareIn):
    try:
        return inference.compare(b.question, b.base_model_id, b.job_id, b.artifact_id)
    except inference.Busy as e:
        err(409, str(e))
    except Exception as e:
        err(400, str(e))


# ------------------------------------------------------------ artifacts (saved AIs)
class ArtifactIn(BaseModel):
    job_id: str
    version_name: str
    description: str = ""
    save_location: str = "local"
    export_format: str = "safetensors"
    target_profile: str | None = None


@app.post("/api/artifacts")
def art_create(b: ArtifactIn):
    j = _job(b.job_id)
    if j["state"] != "completed" or not (JOBS / b.job_id / "adapter").exists():
        err(409, "Finish a training run before saving.")
    aid = db.new_id()
    dst = AI_DIR / aid
    shutil.copytree(JOBS / b.job_id / "adapter", dst / "adapter")
    ev = JOBS / b.job_id / "evaluation.json"
    if ev.exists():
        shutil.copy2(ev, dst / "evaluation.json")
    lic = next((m["license"] for m in SUPPORTED_MODELS if m["model_id"] == j["model_id"]), "")
    size = sum(f.stat().st_size for f in (dst / "adapter").rglob("*") if f.is_file())
    manifest = {"description": b.description, "method": (j["metrics"] or {}).get("method") or j["config"].get("method"),
                "training_config": j["config"], "dataset_id": j["dataset_id"], "evaluation_metrics": j["metrics"],
                "save_location": b.save_location, "target_profile": b.target_profile, "job_id": b.job_id,
                "adapter_bytes": size, "base_license": lic, "project_name": j["project_name"],
                "train_seconds": (j["metrics"] or {}).get("train_seconds")}
    db.run("INSERT INTO artifacts(id,job_id,version_name,base_model_id,created_at,manifest) VALUES(?,?,?,?,?,?)",
           (aid, b.job_id, b.version_name, j["model_id"], time.time(), json.dumps(manifest)))
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"artifact_id": aid, "path": str(dst)}


@app.get("/api/artifacts")
def art_list():
    rows = db.q("SELECT * FROM artifacts ORDER BY created_at DESC")
    for r in rows:
        r["manifest"] = db.jload(r["manifest"], {})
    return rows


@app.delete("/api/artifacts/{aid}")
def art_delete(aid: str):
    db.run("DELETE FROM artifacts WHERE id=?", (aid,))
    shutil.rmtree(AI_DIR / aid, ignore_errors=True)
    return {"ok": True}


class ArtifactPatch(BaseModel):
    description: str | None = None
    version_name: str | None = None


@app.patch("/api/artifacts/{aid}")
def art_patch(aid: str, b: ArtifactPatch):
    a = db.one("SELECT * FROM artifacts WHERE id=?", (aid,))
    if not a:
        err(404, "Not found.")
    m = db.jload(a["manifest"], {})
    if b.description is not None:
        m["description"] = b.description
    db.run("UPDATE artifacts SET manifest=?, version_name=? WHERE id=?",
           (json.dumps(m), b.version_name or a["version_name"], aid))
    return {"ok": True}


@app.get("/api/artifacts/{aid}/files")
def art_files(aid: str):
    d = AI_DIR / aid / "adapter"
    return [{"name": f.name, "size": f.stat().st_size} for f in sorted(d.glob("*")) if f.is_file()]


@app.get("/api/artifacts/{aid}/evaluation")
def art_eval(aid: str):
    p = AI_DIR / aid / "evaluation.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


# ------------------------------------------------------------ export
class ExportIn(BaseModel):
    format: str = "adapter"


@app.post("/api/artifacts/{aid}/export")
def exp_start(aid: str, b: ExportIn):
    try:
        return {"export_id": exporter.start(aid, b.format)}
    except (ValueError, RuntimeError) as e:
        err(409, str(e))


@app.get("/api/exports/{eid}")
def exp_get(eid: str):
    v = exporter.view(eid)
    if not v:
        err(404, "Export not found.")
    return v


@app.post("/api/exports/{eid}/cancel")
def exp_cancel(eid: str):
    return {"ok": exporter.cancel(eid)}


@app.post("/api/exports/{eid}/reveal")
def exp_reveal(eid: str):
    return {"ok": exporter.reveal(eid)}


@app.get("/api/exports/{eid}/download")
def exp_download(eid: str):
    s = exporter.STATE.get(eid)
    if not s or s["state"] != "done" or not Path(s["file"]).exists():
        err(404, "The export isn't ready.")
    return FileResponse(s["file"], filename=Path(s["file"]).name, media_type="application/zip")


# ------------------------------------------------------------ library
@app.get("/api/library")
def lib_list():
    out = []
    for it in library.list_items():
        it["size_bytes"] = it.get("bytes") or sum(f.get("size", 0) for f in it.get("files", []))
        out.append(it)
    try:
        free = round(shutil.disk_usage(str(config.MODELS)).free / 1073741824, 1)
    except OSError:
        free = 0
    return {"root": str(config.MODELS), "items": out, "free_gb": free}


@app.get("/api/library/hf-files")
def lib_files(repo: str, token: str | None = None):
    try:
        info = library.hf_info(repo, token)
    except Exception as e:
        err(502, f"Couldn't read the file list: {e}")
    return {"gated": bool(info.get("gated")), "private": bool(info.get("private")),
            "files": [{"rfilename": s["rfilename"], "size": s.get("size") or 0}
                      for s in library.pick_files(info.get("siblings") or [])]}


class DlIn(BaseModel):
    meta: dict
    files: list[dict]
    sub: str | None = None
    token: str | None = None


@app.post("/api/library/download")
def lib_dl_start(b: DlIn):
    try:
        library.start_download(b.meta, b.files, b.sub, b.token)
    except RuntimeError as e:
        err(409, str(e))
    return library.DL["job"].view()


@app.get("/api/library/download")
def lib_dl_state():
    j = library.DL["job"]
    return j.view() if j else None


@app.post("/api/library/download/{action}")
def lib_dl_action(action: str):
    j = library.DL["job"]
    if not j:
        err(404, "No download.")
    if action == "pause":
        j.pause()
    elif action == "resume":
        if j.state in ("paused", "error"):
            j.start()
    elif action == "cancel":
        j.cancel()
        library.DL["job"] = None
    elif action == "close":
        library.DL["job"] = None
    return j.view()


@app.delete("/api/library")
def lib_delete(path: str):
    try:
        library.delete(path)
    except ValueError:
        err(400, "Bad path.")
    inference.unload()
    return {"ok": True}


# ------------------------------------------------------------ UI
# In the packaged exe the ui folder sits next to the bundled modules (sys._MEIPASS); from source it is ../ui.
_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
UI = _ROOT / "ui"
if (UI / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(UI), html=True), name="ui")
else:
    @app.get("/", include_in_schema=False)
    def ui_missing():
        return JSONResponse({"detail": f"The AOMe user interface files were not found at {UI}. "
                                       "Rebuild the app or re-extract the download."}, status_code=500)
