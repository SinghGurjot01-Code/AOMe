"""LoRA / QLoRA fine-tuning engine. Real training loop with live progress, real checkpoints,
held-out evaluation against the base model, and sample answer comparison."""
import contextlib
import json
import math
import random
import re
import shutil
import threading
import time
import traceback
from html import escape
from pathlib import Path

from . import datasets, db, hardware, library, llm
from .config import JOBS, SUPPORTED_MODELS
from .llm import Cancelled

RUNNERS = {}          # job_id -> Runner
_active_lock = threading.Lock()


class Runner:
    def __init__(self, job_id):
        self.job_id = job_id
        self.cancel = threading.Event()
        self.stop_at = None
        self.thread = None
        self.model = None
        self.prog = {}            # live progress shown in the UI: phase, first_run, step_sec, eta_sec ...


def active_job():
    for jid, r in RUNNERS.items():
        if r.thread and r.thread.is_alive():
            return jid
    return None


# ---------------------------------------------------------------- estimate
def params_b_for(model_id):
    for m in SUPPORTED_MODELS:
        if m["model_id"] == model_id:
            return library.parse_params_b(m["parameters"])
    for it in library.list_items():
        if it.get("id") == model_id and it.get("params"):
            v = library.parse_params_b(it["params"])
            if v:
                return v
    m = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])(?![a-zA-Z])", model_id.split("/")[-1])
    if m:
        return float(m.group(1)) * (1 if m.group(2).lower() == "b" else 0.001)
    return 1.0


def estimate(model_id, sample_count, cfg):
    hw = hardware.snapshot()
    p = params_b_for(model_id)
    method = cfg.get("method", "lora")
    q4 = method == "qlora"
    bs = int(cfg.get("batch_size", 2))
    seq = int(cfg.get("max_seq_length", 2048))
    epochs = int(cfg.get("epochs", 3))
    warnings = []
    gpu = hw["cuda_available"]
    if gpu:
        bpp = 0.57 if q4 else 2.0
        vram = p * 1024 * bpp * 1.15 + 1200 + 200 * bs * (seq / 1024) * max(1.0, p ** 0.5)
        ram = p * 1024 * 2 * 0.6 + 2500
        compatible = hw["vram_total_mb"] >= vram * 0.95
        if not compatible:
            warnings.append(f"Needs about {vram / 1024:.1f} GB of GPU memory but this GPU has {hw['vram_total_mb'] / 1024:.1f} GB. "
                            f"Try QLoRA, a smaller model, batch size 1 or a shorter context.")
        elif hw["vram_total_mb"] < vram * 1.1:
            warnings.append("Tight fit. Close other GPU programs before training.")
        if q4 and not llm.bnb_available():
            warnings.append("4-bit (QLoRA) needs the bitsandbytes package; training will fall back to regular LoRA.")
        tf = 8 if hw["vram_total_mb"] <= 8192 else 14 if hw["vram_total_mb"] <= 12288 else 22 if hw["vram_total_mb"] <= 16384 \
            else 35 if hw["vram_total_mb"] <= 24576 else 60
    else:
        vram = 0
        ram = p * 1024 * 4 * 1.3 + 2000
        compatible = hw["ram_total_mb"] >= ram
        warnings.append("No NVIDIA GPU detected. Training will run on the CPU and is much slower. "
                        "Small models (under 1B) are recommended.")
        if not compatible:
            warnings.append(f"Needs about {ram / 1024:.1f} GB of RAM but this PC has {hw['ram_total_mb'] / 1024:.1f} GB.")
        tf = max(0.05, 0.03 * hw["cpu_threads"])
    tokens = max(1, sample_count) * 0.8 * 180 * epochs
    minutes = 4 * p * 1e9 * tokens / (tf * 1e12) / 60 + 1.0
    disk = 0.5 if library.find_local(model_id) else p * 2.1 + 0.5
    if hw["disk_free_gb"] < disk:
        compatible = False
        warnings.append(f"Not enough free disk space: need about {disk:.1f} GB, have {hw['disk_free_gb']} GB.")
    return {"required_vram_mb": int(vram), "required_ram_mb": int(ram), "required_disk_gb": round(disk, 1),
            "estimated_duration_minutes": round(minutes, 1), "compatible": bool(compatible), "warnings": warnings}


# ---------------------------------------------------------------- job control
def progress(job_id, **kw):
    """Merge live progress fields (phase timing, per-step speed, time left) and store them with the job."""
    r = RUNNERS.get(job_id)
    if r is None:
        return
    r.prog.update(kw, updated=time.time())
    db.run("UPDATE jobs SET progress=? WHERE id=?", (json.dumps(r.prog), job_id))


@contextlib.contextmanager
def heartbeat(job_id, label, every=8.0):
    """Long blocking calls (loading weights, evaluation) can't report progress themselves. While one runs,
    write a log line every few seconds so the person can see the engine is alive and not frozen."""
    stop = threading.Event()
    t0 = time.time()

    def beat():
        while not stop.wait(every):
            db.add_event(job_id, "log", f"{label}… still working ({int(time.time() - t0)}s)")
            progress(job_id)
    th = threading.Thread(target=beat, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()


def set_state(job_id, state, message=None, **fields):
    progress(job_id, phase=state, phase_started=time.time())
    cols = ["state=?"]
    args = [state]
    for k, v in fields.items():
        cols.append(f"{k}=?")
        args.append(v)
    args.append(job_id)
    db.run(f"UPDATE jobs SET {', '.join(cols)} WHERE id=?", args)
    if message:
        db.add_event(job_id, "state", message)


def start(project_name, dataset_id, model_id, cfg):
    if active_job():
        raise RuntimeError("A training job is already running.")
    if not db.one("SELECT id FROM datasets WHERE id=?", (dataset_id,)):
        raise ValueError("Dataset not found. Process your data again.")
    if not llm.torch_ok():
        raise ValueError("Training needs PyTorch, which isn't installed with this copy of AOMe. "
                         "Reinstall AOMe or install PyTorch (see the README).")
    if not library.find_local(model_id):
        raise ValueError("That base model isn't downloaded on this PC. Download it from Explore Models, "
                         "or pick one from your Library.")
    jid = db.new_id()
    (JOBS / jid).mkdir(parents=True, exist_ok=True)
    db.run("INSERT INTO jobs(id,project_name,dataset_id,model_id,state,created_at,config) VALUES(?,?,?,?,?,?,?)",
           (jid, project_name, dataset_id, model_id, "queued", time.time(), json.dumps(cfg)))
    r = Runner(jid)
    RUNNERS[jid] = r
    r.thread = threading.Thread(target=_run, args=(r,), daemon=True)
    r.thread.start()
    return jid


def cancel(job_id):
    r = RUNNERS.get(job_id)
    if r and r.thread and r.thread.is_alive():
        r.cancel.set()
        set_state(job_id, "cancelling", "Stopping…")
        return True
    return False


def stop_at(job_id, step):
    r = RUNNERS.get(job_id)
    if not r or not r.thread or not r.thread.is_alive():
        return False
    r.stop_at = int(step) if step else None
    if step:
        db.add_event(job_id, "info", f"Will stop at checkpoint step {int(step)} and evaluate the model from there.")
    else:
        db.add_event(job_id, "info", "Stop request removed. Training continues.")
    return True


# ---------------------------------------------------------------- data
def _encode(tok, msgs, max_len):
    prompt = llm.build_prompt(tok, msgs[:-1])
    ans = msgs[-1]["content"] + (tok.eos_token or "")
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(ans, add_special_tokens=False)["input_ids"]
    if len(p) + len(a) > max_len:
        keep_p = max(max_len // 2, max_len - len(a))
        p = p[-keep_p:]
        a = a[: max_len - len(p)]
    return {"ids": p + a, "labels": [-100] * len(p) + a}


def _collate(exs, pad_id, device):
    import torch
    n = max(len(e["ids"]) for e in exs)
    ids = torch.full((len(exs), n), pad_id, dtype=torch.long)
    lab = torch.full((len(exs), n), -100, dtype=torch.long)
    att = torch.zeros((len(exs), n), dtype=torch.long)
    for i, e in enumerate(exs):
        L = len(e["ids"])
        ids[i, :L] = torch.tensor(e["ids"])
        lab[i, :L] = torch.tensor(e["labels"])
        att[i, :L] = 1
    return ids.to(device), att.to(device), lab.to(device)


def _eval_loss(model, tok, exs, bs, device, amp, cancel):
    import torch
    import torch.nn.functional as F
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(exs), bs):
            if cancel.is_set():
                raise Cancelled()
            ids, att, lab = _collate(exs[i:i + bs], tok.pad_token_id, device)
            with torch.autocast(device_type=device, dtype=amp, enabled=amp is not None):
                lg = model(input_ids=ids, attention_mask=att).logits
            sh = lg[:, :-1].float().reshape(-1, lg.shape[-1])
            lb = lab[:, 1:].reshape(-1)
            tot += F.cross_entropy(sh, lb, ignore_index=-100, reduction="sum").item()
            cnt += int((lb != -100).sum().item())
    return tot / max(1, cnt)


def _html(t):
    parts = [p.strip() for p in re.split(r"\n{2,}", t.strip()) if p.strip()] or [""]
    return "".join("<p>" + escape(p).replace("\n", "<br>") + "</p>" for p in parts)


def _evaluate(model, tok, test_msgs, bs, device, amp, cancel, jid, cpu):
    """Compare adapter vs base on held-out records. Returns (metrics, samples)."""
    import torch
    max_len = 1024
    exs = [_encode(tok, m, max_len) for m in test_msgs]
    db.add_event(jid, "log", f"Measuring loss on {len(exs)} held-out record(s)…")
    tuned = _eval_loss(model, tok, exs, bs, device, amp, cancel)
    with model.disable_adapter():
        base = _eval_loss(model, tok, exs, bs, device, amp, cancel)
    bp, tp = math.exp(min(base, 20)), math.exp(min(tuned, 20))
    n_gen = min(len(test_msgs), 4 if cpu else 6)
    new_tok = 100 if cpu else 160
    model.config.use_cache = True
    samples, sa_all, sb_all = [], [], []
    model.eval()
    for i in range(n_gen):
        if cancel.is_set():
            raise Cancelled()
        m = test_msgs[i]
        prompt, ref = m[:-1], m[-1]["content"]
        db.add_event(jid, "log", f"Generating comparison answer {i + 1}/{n_gen}…")
        a = llm.generate(model, tok, prompt, new_tok, 0.0, cancel=cancel)
        with model.disable_adapter():
            b = llm.generate(model, tok, prompt, new_tok, 0.0, cancel=cancel)
        sa, sb = llm.f1(a, ref), llm.f1(b, ref)
        sa_all.append(sa)
        sb_all.append(sb)
        q = prompt[-1]["content"]
        samples.append({"t": (q[:26] + "…") if len(q) > 27 else q, "q": q, "b": _html(b), "a": _html(a),
                        "sb": sb, "sa": sa, "relevance": sa, "reference": ref})
    mean = lambda v: round(sum(v) / len(v), 1) if v else 0.0
    metrics = {"test_sample_count": len(exs), "base_eval_loss": round(base, 3), "custom_eval_loss": round(tuned, 3),
               "base_perplexity": round(bp, 2), "custom_perplexity": round(tp, 2),
               "perplexity_reduction_pct": round((1 - tp / bp) * 100, 1),
               "answer_relevance_pct": mean(sa_all), "base_answer_relevance_pct": mean(sb_all),
               "compared_answers": n_gen}
    return metrics, samples


# ---------------------------------------------------------------- the run
def _run(r: Runner):
    jid = r.job_id
    job = db.one("SELECT * FROM jobs WHERE id=?", (jid,))
    cfg = db.jload(job["config"], {})
    ev = lambda m, t="log", d=None: db.add_event(jid, t, m, d)
    try:
        _execute(r, job, cfg, ev)
    except Cancelled:
        set_state(jid, "cancelled", "Training stopped.", finished_at=time.time())
    except Exception as e:
        msg = str(e) or e.__class__.__name__
        if "out of memory" in msg.lower():
            msg = ("The GPU ran out of memory. Try QLoRA, a smaller model, batch size 1, or a shorter context length.")
        db.add_event(jid, "log", "ERROR: " + msg)
        db.add_event(jid, "log", traceback.format_exc().strip().splitlines()[-1])
        set_state(jid, "failed", None, error_message=msg, finished_at=time.time())
    finally:
        r.model = None
        llm.free_memory()


def _execute(r, job, cfg, ev):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import get_cosine_schedule_with_warmup

    jid = r.job_id
    jdir = JOBS / jid
    info = llm.device_info()
    device, amp = info["device"], info["amp"]
    cpu = device == "cpu"
    from . import inference
    inference.unload()

    set_state(jid, "validating", "Checking dataset and configuration…", started_at=time.time())
    train_m = datasets.load_split(job["dataset_id"], "train")
    val_m = datasets.load_split(job["dataset_id"], "val")
    test_m = datasets.load_split(job["dataset_id"], "test")
    if not train_m:
        raise ValueError("The dataset has no training records.")
    eval_m = test_m or val_m or train_m[-max(1, len(train_m) // 10):]
    if not test_m:
        ev("No held-out test split; using validation/training records for evaluation.")
    ev(f"Device: {hardware.snapshot()['device_name']} ({'CUDA' if not cpu else 'CPU'})")

    method = cfg.get("method", "lora")
    q4 = method == "qlora"
    if q4 and not llm.bnb_available():
        ev("bitsandbytes/CUDA not available: using regular LoRA instead of QLoRA.")
        q4 = False
        method = "lora"

    # Only models already in the library are used; nothing is downloaded behind the person's back.
    path = library.find_local(job["model_id"])
    if not path:
        raise ValueError("The base model isn't in your library any more. Download it again from Explore Models.")
    fr = library.first_run_info(job["model_id"])
    progress(jid, first_run=fr["first_run"], first_run_min=fr["minutes"], model=path.name)
    set_state(jid, "downloading_model", "Loading base model…")
    ev(f"Using downloaded model: {path.name}")
    if fr["first_run"]:
        ev(f"First run of this model on this PC: setting it up can take about {fr['minutes']} min. "
           f"This only happens once, later runs start much faster.")
    with heartbeat(jid, "Loading base model"):
        tok = llm.load_tokenizer(path)
        ev("Loading weights…")
        model = llm.load_base(path, quant4=q4)
    library.mark_used(path)
    r.model = model
    ev("Model loaded.")

    set_state(jid, "preparing_data", "Tokenizing training data…")
    max_len = max(64, min(8192, int(cfg.get("max_seq_length", 1024))))
    bs = max(1, int(cfg.get("batch_size", 2)))
    accum = max(1, int(cfg.get("gradient_accumulation_steps", 4)))
    epochs = max(1, int(cfg.get("epochs", 3)))
    lr = float(cfg.get("learning_rate", 2e-4))
    ck_every = max(1, int((cfg.get("extra_params") or {}).get("checkpoint_every") or cfg.get("checkpoint_every") or 50))
    train_ex = [_encode(tok, m, max_len) for m in train_m]
    accum = max(1, min(accum, math.ceil(len(train_ex) / bs)))
    steps_per_epoch = max(1, math.ceil(len(train_ex) / (bs * accum)))
    total = steps_per_epoch * epochs
    db.run("UPDATE jobs SET total_steps=? WHERE id=?", (total, jid))
    ev(f"{len(train_ex)} training records · batch {bs} × accumulation {accum} · {total} steps over {epochs} epoch(s)")

    gc_on = cfg.get("gradient_checkpointing")
    if gc_on is None:
        gc_on = params_b_for(job["model_id"]) >= 1.0
    if q4:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(gc_on))
    elif not cpu and gc_on:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=int(cfg.get("lora_r", 8)), lora_alpha=int(cfg.get("lora_alpha", 16)),
                      lora_dropout=float(cfg.get("lora_dropout", 0.05)), target_modules="all-linear",
                      task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, lcfg)
    r.model = model
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ev(f"Training {trainable / 1e6:.2f}M adapter parameters ({'QLoRA 4-bit' if q4 else 'LoRA'}).")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(total * 0.05)), total)
    scaler = torch.amp.GradScaler("cuda") if (amp == torch.float16 and not cpu) else None
    rng = random.Random(int(cfg.get("seed", 42)))

    set_state(jid, "training", "Training started.")
    model.train()
    model.config.use_cache = False
    step, t0, last_loss = 0, time.time(), None
    step_sec = None                                   # smoothed seconds per optimizer step
    ck_dir = jdir / "checkpoints"
    log_every = max(1, total // 40)
    stopped_at = None
    done = False
    for epoch in range(epochs):
        order = list(range(len(train_ex)))
        rng.shuffle(order)
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        for gi in range(0, len(batches), accum):
            t_step = time.time()
            group = batches[gi:gi + accum]
            ntok = sum(sum(1 for x in train_ex[i]["labels"][1:] if x != -100) for b in group for i in b) or 1
            loss_sum = 0.0
            for b in group:
                if r.cancel.is_set():
                    raise Cancelled()
                ids, att, lab = _collate([train_ex[i] for i in b], tok.pad_token_id, device)
                with torch.autocast(device_type=device, dtype=amp, enabled=amp is not None):
                    lg = model(input_ids=ids, attention_mask=att).logits
                l = F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1),
                                    ignore_index=-100, reduction="sum")
                loss_sum += l.item()
                (scaler.scale(l / ntok) if scaler else (l / ntok)).backward()
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            last_loss = loss_sum / ntok
            dt = time.time() - t_step
            step_sec = dt if step_sec is None else 0.35 * dt + 0.65 * step_sec
            db.run("UPDATE jobs SET current_step=?, loss=? WHERE id=?", (step, last_loss, jid))
            progress(jid, step_sec=round(step_sec, 3), eta_sec=round(max(0, total - step) * step_sec, 1),
                     last_step_at=time.time())
            if step % log_every == 0 or step == 1 or step == total:
                el = time.time() - t0
                ev(f"Epoch {epoch + 1}/{epochs} | Step {step}/{total} | Loss: {last_loss:.3f} | {step / el:.2f} steps/s")
            if step % ck_every == 0 and step < total:
                d = ck_dir / f"step-{step}"
                model.save_pretrained(str(d))
                llm.fix_adapter_config(d, job["model_id"])
                db.add_event(jid, "checkpoint", f"Checkpoint saved at step {step}", {"step": step, "loss": last_loss})
            if r.stop_at and step >= r.stop_at:
                stopped_at = step
                done = True
                break
        if done:
            break

    final_step = step
    model.eval()
    model.config.use_cache = True
    if stopped_at:
        ev(f"Stopped at checkpoint step {stopped_at}. Evaluating the model from this point.")
    adapter = jdir / "adapter"
    shutil.rmtree(adapter, ignore_errors=True)
    model.save_pretrained(str(adapter))
    tok.save_pretrained(str(adapter))
    llm.fix_adapter_config(adapter, job["model_id"])

    set_state(jid, "evaluating", "Comparing against the base model on held-out data…")
    progress(jid, eta_sec=None)
    with heartbeat(jid, "Evaluating"):
        metrics, samples = _evaluate(model, tok, eval_m, bs, device, amp, r.cancel, jid, cpu)
    metrics.update({"checkpoint_step": final_step, "total_steps": total, "stopped_early": bool(stopped_at),
                    "train_seconds": round(time.time() - t0, 1), "method": method,
                    "used_validation_fallback": not bool(test_m)})
    (jdir / "evaluation.json").write_text(json.dumps(samples), encoding="utf-8")
    (jdir / "job.json").write_text(json.dumps({"base_model_id": job["model_id"], "base_path": str(path),
                                               "config": cfg, "method": method}, indent=2), encoding="utf-8")
    db.add_event(jid, "eval_result", "Evaluation finished.", metrics)

    set_state(jid, "packaging", "Writing adapter…")
    db.run("UPDATE jobs SET metrics=?, current_step=? WHERE id=?", (json.dumps(metrics), final_step, jid))
    set_state(jid, "completed", "Training complete.", finished_at=time.time())


# ---------------------------------------------------------------- re-evaluate another checkpoint
def select_checkpoint(job_id, step):
    """Make a saved checkpoint the job's model and re-run the evaluation on it (background)."""
    job = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
    if not job or job["state"] != "completed":
        raise ValueError("The job has to be finished first.")
    src = JOBS / job_id / "checkpoints" / f"step-{int(step)}"
    if not src.exists():
        raise ValueError(f"No checkpoint was saved at step {step}.")
    if active_job():
        raise RuntimeError("Another job is running.")
    r = Runner(job_id)
    RUNNERS[job_id] = r

    def work():
        try:
            from peft import PeftModel
            from . import inference
            inference.unload()
            set_state(job_id, "evaluating", f"Evaluating checkpoint step {step}…")
            cfg = db.jload(job["config"], {})
            info = llm.device_info()
            meta = json.loads((JOBS / job_id / "job.json").read_text())
            tok = llm.load_tokenizer(JOBS / job_id / "adapter")
            base = llm.load_base(meta["base_path"])
            model = PeftModel.from_pretrained(base, str(src))
            model.eval()
            test_m = datasets.load_split(job["dataset_id"], "test") or datasets.load_split(job["dataset_id"], "val") \
                or datasets.load_split(job["dataset_id"], "train")[-3:]
            metrics, samples = _evaluate(model, tok, test_m, max(1, int(cfg.get("batch_size", 2))),
                                         info["device"], info["amp"], r.cancel, job_id, info["device"] == "cpu")
            old = db.jload(job["metrics"], {})
            metrics.update({"checkpoint_step": int(step), "total_steps": old.get("total_steps"),
                            "stopped_early": int(step) != old.get("total_steps"), "method": old.get("method"),
                            "train_seconds": old.get("train_seconds")})
            adapter = JOBS / job_id / "adapter"
            for f in src.iterdir():
                shutil.copy2(f, adapter / f.name)
            llm.fix_adapter_config(adapter, job["model_id"])
            (JOBS / job_id / "evaluation.json").write_text(json.dumps(samples), encoding="utf-8")
            db.add_event(job_id, "eval_result", f"Evaluation of checkpoint step {step} finished.", metrics)
            db.run("UPDATE jobs SET metrics=?, current_step=? WHERE id=?", (json.dumps(metrics), int(step), job_id))
            set_state(job_id, "completed", f"Using checkpoint step {step}.")
        except Exception as e:
            db.add_event(job_id, "log", "ERROR: " + str(e))
            set_state(job_id, "completed", None)
        finally:
            llm.free_memory()
    r.thread = threading.Thread(target=work, daemon=True)
    r.thread.start()
