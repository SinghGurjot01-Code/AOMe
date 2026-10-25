"""Chat / compare against base models and trained adapters. One model stays loaded at a time."""
import contextlib
import json
import threading
import time

from . import db, library, llm, trainer
from .config import AI_DIR, JOBS

_lock = threading.RLock()
_gen = threading.Lock()
S = {"base_id": None, "tok": None, "model": None, "peft": False, "adapters": set()}


class Busy(Exception):
    pass


def unload():
    with _lock:
        S.update(base_id=None, tok=None, model=None, peft=False, adapters=set())
    llm.free_memory()


def resolve(model_type="base", artifact_id=None, job_id=None, base_model_id=None):
    """-> (base_id, adapter_dir or None, label)"""
    if model_type == "custom" and artifact_id:
        a = db.one("SELECT * FROM artifacts WHERE id=?", (artifact_id,))
        if not a:
            raise ValueError("That saved AI no longer exists.")
        return a["base_model_id"], AI_DIR / artifact_id / "adapter", a["version_name"]
    if model_type == "custom" and job_id:
        jd = JOBS / job_id
        if not (jd / "job.json").exists():
            raise ValueError("That training run has no finished model yet.")
        meta = json.loads((jd / "job.json").read_text())
        return meta["base_model_id"], jd / "adapter", "Your model"
    if not base_model_id:
        raise ValueError("Pick a model first.")
    return base_model_id, None, base_model_id.split("/")[-1]


def _load(base_id, adapter_dir, key):
    from peft import PeftModel
    with _lock:
        if S["base_id"] != base_id:
            unload()
            path = library.find_local(base_id)
            if not path:
                raise RuntimeError(f"The base model {base_id} isn't downloaded yet. Get it from the Library page.")
            S["tok"] = llm.load_tokenizer(path)
            S["model"] = llm.load_base(path)
            library.mark_used(path)
            S["base_id"], S["peft"], S["adapters"] = base_id, False, set()
            S["model"].eval()
        if adapter_dir is not None:
            if not S["peft"]:
                S["model"] = PeftModel.from_pretrained(S["model"], str(adapter_dir), adapter_name=key)
                S["peft"] = True
                S["adapters"].add(key)
            elif key not in S["adapters"]:
                S["model"].load_adapter(str(adapter_dir), adapter_name=key)
                S["adapters"].add(key)
            S["model"].set_adapter(key)
            ctx = contextlib.nullcontext()
        else:
            ctx = S["model"].disable_adapter() if S["peft"] else contextlib.nullcontext()
        S["model"].eval()
        S["model"].config.use_cache = True
        return S["tok"], S["model"], ctx


def _guard():
    if trainer.active_job():
        raise Busy("Training is running. Chat is available again when it finishes (the GPU is in use).")


def chat_start(messages, system, model_type, artifact_id, job_id, base_model_id, temperature, max_new_tokens):
    """Start a reply in a worker thread. Returns (queue, stop_event); the queue yields dicts, then None.
    The worker owns the generation lock, so a dropped client can never leave it stuck."""
    import queue
    q = queue.Queue()
    stop = threading.Event()
    try:
        _guard()
        base_id, adapter, label = resolve(model_type, artifact_id, job_id, base_model_id)
        if not _gen.acquire(blocking=False):
            raise Busy("Another reply is still being generated.")
    except Exception as e:
        q.put({"error": str(e)})
        q.put(None)
        return q, stop

    def work():
        try:
            fr = library.first_run_info(base_id) if S["base_id"] != base_id else {"first_run": False, "minutes": 0}
            q.put({"status": "loading", "model": label, "first_run": fr["first_run"], "minutes": fr["minutes"]})
            key = artifact_id or job_id or "custom"
            tok, model, ctx = _load(base_id, adapter, key)
            msgs = ([{"role": "system", "content": system}] if system and system.strip() else []) + messages[-12:]
            n, t0 = 0, time.time()
            q.put({"status": "generating", "model": label})
            with ctx:
                for piece in llm.stream_generate(model, tok, msgs, int(max_new_tokens), float(temperature), 0.9, stop):
                    n += 1
                    q.put({"t": piece})
            q.put({"done": True, "pieces": n, "seconds": round(time.time() - t0, 2), "model": label})
        except Exception as e:
            msg = str(e)
            if "out of memory" in msg.lower():
                msg = "The GPU ran out of memory for this reply. Try a shorter message or close other GPU programs."
            q.put({"error": msg})
        finally:
            _gen.release()
            q.put(None)
    threading.Thread(target=work, daemon=True).start()
    return q, stop


def compare(question, base_model_id, job_id=None, artifact_id=None, max_new_tokens=160):
    from html import escape
    import re
    _guard()
    if not _gen.acquire(blocking=False):
        raise Busy("Another reply is still being generated.")
    try:
        base_id, adapter, _ = resolve("custom", artifact_id, job_id, base_model_id)
        tok, model, _ctx = _load(base_id, adapter, artifact_id or job_id or "custom")
        msgs = [{"role": "user", "content": question}]
        a = llm.generate(model, tok, msgs, max_new_tokens, 0.0)
        with model.disable_adapter():
            b = llm.generate(model, tok, msgs, max_new_tokens, 0.0)
    finally:
        _gen.release()
    h = lambda t: "".join("<p>" + escape(p).replace("\n", "<br>") + "</p>" for p in re.split(r"\n{2,}", t.strip()) if p.strip())
    return {"question": question, "a": h(a), "b": h(b), "relevance": None}
