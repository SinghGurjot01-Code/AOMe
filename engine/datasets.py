"""Dataset validation, normalisation and train/val/test splitting.

Accepted record shapes (one JSON object per line):
  {"messages":[{"role":"system|user|assistant","content":"..."}, ...]}
  {"instruction"|"prompt"|"question"|"input"|"user": "...", "response"|"output"|"answer"|"completion"|"assistant": "..."}
"""
import json
import random
import shutil
import time
from pathlib import Path

from . import db
from .config import DATASETS, MAX_UPLOAD_MB

Q_KEYS = ["question", "prompt", "instruction", "input", "user", "query", "q"]
A_KEYS = ["answer", "response", "output", "completion", "assistant", "reply", "a"]
ROLES = {"system", "user", "assistant"}


def _pick(o, keys):
    for k in keys:
        v = o.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def normalise(obj):
    """Return a messages list or None."""
    if not isinstance(obj, dict):
        return None
    msgs = obj.get("messages")
    if isinstance(msgs, list):
        out = []
        for m in msgs:
            if not isinstance(m, dict):
                return None
            role, content = m.get("role"), m.get("content")
            if role not in ROLES or not isinstance(content, str) or not content.strip():
                return None
            out.append({"role": role, "content": content.strip()})
        if any(m["role"] == "user" for m in out) and out and out[-1]["role"] == "assistant":
            return out
        return None
    q, a = _pick(obj, Q_KEYS), _pick(obj, A_KEYS)
    if q and a:
        inst = obj.get("system")
        out = [{"role": "system", "content": inst.strip()}] if isinstance(inst, str) and inst.strip() else []
        return out + [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return None


def parse_file(name: str, data: bytes):
    """Return (records, errors, warnings) for one uploaded file."""
    errs, warns, recs = [], [], []
    if len(data) > MAX_UPLOAD_MB * 1048576:
        return [], [f"[{name}] Larger than {MAX_UPLOAD_MB} MB."], []
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [], [f"[{name}] Not valid UTF-8 text."], []
    if name.lower().endswith(".json"):
        try:
            j = json.loads(text)
            objs = j if isinstance(j, list) else (j.get("data") or j.get("items") or j.get("records") or [j])
        except Exception as e:
            return [], [f"[{name}] Invalid JSON: {str(e)[:80]}"], []
    else:
        objs = []
        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                objs.append(json.loads(line))
            except Exception:
                errs.append(f"[{name}] Line {i} is not valid JSON.")
                if len(errs) >= 5:
                    errs.append(f"[{name}] More lines have problems.")
                    break
    bad = 0
    for o in objs:
        m = normalise(o)
        if m:
            recs.append(m)
        else:
            bad += 1
    if bad and recs:
        warns.append(f"[{name}] {bad} record(s) skipped (need a user message followed by an assistant reply).")
    if not recs and not errs:
        errs.append(f"[{name}] No usable chat records found.")
    return recs, errs, warns


def validate(files):
    """files: list of (name, bytes). Returns the validation payload."""
    allrecs, errors, warnings = [], [], []
    for name, data in files:
        r, e, w = parse_file(name, data)
        allrecs += r
        errors += e
        warnings += w
    seen, uniq = set(), []
    for r in allrecs:
        k = json.dumps(r, sort_keys=True)
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    if len(uniq) < len(allrecs):
        warnings.append(f"{len(allrecs) - len(uniq)} duplicate record(s) will be removed.")
    if not errors:
        if len(uniq) < 3:
            errors.append("Need at least 3 chat records to train and test a model.")
        elif len(uniq) < 50:
            warnings.append(f"Only {len(uniq)} records. Small datasets work for a test run; 100+ gives better results.")
    toks = sum(len(m["content"]) for r in uniq for m in r) // 4
    return uniq, {"is_valid": not errors, "sample_count": len(uniq), "errors": errors, "warnings": warnings,
                  "token_count_estimate": toks}


def split(recs, seed=42):
    r = recs[:]
    random.Random(seed).shuffle(r)
    n = len(r)
    if n < 10:
        te, va = 1, 0
    else:
        te = va = max(1, round(n * 0.1))
    return r[: n - va - te], r[n - va - te: n - te], r[n - te:]


def create(files, name):
    recs, v = validate(files)
    if v["errors"]:
        return None, v
    tr, va, te = split(recs)
    did = db.new_id()
    d = DATASETS / did
    d.mkdir(parents=True, exist_ok=True)
    for fn, part in (("train.jsonl", tr), ("val.jsonl", va), ("test.jsonl", te)):
        with open(d / fn, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({"messages": r}, ensure_ascii=False) + "\n")
    manifest = {
        "dataset_id": did, "name": name, "task": "chat", "created_at": time.time(),
        "source_files": [{"filename": n, "size_bytes": len(b), "file_type": Path(n).suffix.lower()} for n, b in files],
        "split_info": {"train_samples": len(tr), "val_samples": len(va), "test_samples": len(te)},
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    db.run("INSERT INTO datasets(id,name,created_at,sample_count,manifest) VALUES(?,?,?,?,?)",
           (did, name, manifest["created_at"], len(recs), json.dumps(manifest)))
    return {"dataset_id": did, "manifest": manifest, "warnings": v["warnings"]}, v


def load_split(dataset_id, part):
    p = DATASETS / dataset_id / f"{part}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l)["messages"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def delete(dataset_id):
    db.run("DELETE FROM datasets WHERE id=?", (dataset_id,))
    shutil.rmtree(DATASETS / dataset_id, ignore_errors=True)
