"""Shared model helpers: device/dtype, loading, prompt building, generation, memory cleanup."""
import contextlib
import gc
import re
import threading

from . import library


class Cancelled(Exception):
    pass


def torch_ok() -> bool:
    """True if PyTorch is installed (checked without importing it, which is slow)."""
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def torch():
    import torch as t
    return t


def device_info():
    t = torch()
    if t.cuda.is_available():
        bf = False
        try:
            bf = t.cuda.is_bf16_supported()
        except Exception:
            pass
        return {"device": "cuda", "dtype": t.bfloat16 if bf else t.float16, "amp": t.bfloat16 if bf else t.float16}
    return {"device": "cpu", "dtype": t.float32, "amp": None}


def _tf_ge(major, minor):
    import transformers
    p = transformers.__version__.split(".")
    return (int(p[0]), int(re.sub(r"\D.*", "", p[1]))) >= (major, minor)


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(path, quant4=False, dtype=None, device=None):
    """Load a causal LM. quant4 needs bitsandbytes + CUDA."""
    from transformers import AutoModelForCausalLM
    t = torch()
    info = device_info()
    dtype = dtype or info["dtype"]
    kw = {"low_cpu_mem_usage": True}
    kw["dtype" if _tf_ge(4, 56) else "torch_dtype"] = dtype
    if quant4:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(str(path), **kw)
    if not quant4:
        model.to(device or info["device"])
    return model


def bnb_available():
    try:
        import bitsandbytes  # noqa
        return torch().cuda.is_available()
    except Exception:
        return False


def fallback_template(msgs, gen=True):
    s = ""
    for m in msgs:
        tag = {"system": "System", "user": "User", "assistant": "Assistant"}[m["role"]]
        s += f"### {tag}:\n{m['content']}\n\n"
    return s + ("### Assistant:\n" if gen else "")


def build_prompt(tok, msgs):
    try:
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    return fallback_template(msgs)


def free_memory():
    gc.collect()
    try:
        t = torch()
        if t.cuda.is_available():
            t.cuda.empty_cache()
            t.cuda.ipc_collect()
    except Exception:
        pass


def generate(model, tok, msgs, max_new_tokens=160, temperature=0.0, top_p=0.9, cancel=None):
    """Blocking generation. Returns the reply text."""
    t = torch()
    prompt = build_prompt(tok, msgs)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if temperature and temperature > 0.01:
        kw.update(do_sample=True, temperature=float(temperature), top_p=top_p)
    else:
        kw.update(do_sample=False)
    if cancel is not None:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _S(StoppingCriteria):
            def __call__(self, input_ids, scores, **k):
                return cancel.is_set()
        kw["stopping_criteria"] = StoppingCriteriaList([_S()])
    with t.no_grad():
        out = model.generate(**enc, **kw)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def stream_generate(model, tok, msgs, max_new_tokens, temperature, top_p, stop_event):
    """Yield text pieces as they are produced."""
    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
    prompt = build_prompt(tok, msgs)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    class _S(StoppingCriteria):
        def __call__(self, input_ids, scores, **k):
            return stop_event.is_set()
    kw = dict(**enc, max_new_tokens=max_new_tokens, streamer=streamer,
              pad_token_id=tok.pad_token_id or tok.eos_token_id,
              stopping_criteria=StoppingCriteriaList([_S()]))
    if temperature and temperature > 0.01:
        kw.update(do_sample=True, temperature=float(temperature), top_p=top_p)
    else:
        kw.update(do_sample=False)
    err = {}

    def work():
        try:
            with torch().no_grad():
                model.generate(**kw)
        except Exception as e:  # make sure the reader unblocks
            err["e"] = e
            streamer.end()
    th = threading.Thread(target=work, daemon=True)
    th.start()
    n = 0
    try:
        for piece in streamer:
            n += 1
            yield piece
    finally:
        stop_event.set()
        th.join(timeout=5)
    if "e" in err:
        raise err["e"]


def f1(a: str, b: str) -> float:
    """Token-overlap F1 (0..100) between a reply and a reference answer."""
    ta = re.findall(r"\w+", a.lower())
    tb = re.findall(r"\w+", b.lower())
    if not ta or not tb:
        return 0.0
    from collections import Counter
    ca, cb = Counter(ta), Counter(tb)
    common = sum((ca & cb).values())
    if not common:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return round(2 * p * r / (p + r) * 100, 1)


def fix_adapter_config(adapter_dir, base_id):
    """Make an adapter portable: point it at the Hugging Face id instead of a local folder."""
    import json
    from pathlib import Path
    p = Path(adapter_dir) / "adapter_config.json"
    if p.exists():
        j = json.loads(p.read_text(encoding="utf-8"))
        j["base_model_name_or_path"] = base_id
        p.write_text(json.dumps(j, indent=2), encoding="utf-8")


def gpu_diagnosis(refresh=False):
    """Explain in plain words why training would (not) use the GPU, and what to do about it."""
    import shutil
    from . import hardware
    if refresh:
        hardware.reset_nvml()
    out = {"ok": False, "gpu_name": None, "torch_installed": torch_ok(), "torch_cuda_build": None,
           "driver_found": bool(hardware._NV), "nvidia_smi": bool(shutil.which("nvidia-smi")),
           "code": "", "title": "", "detail": "", "steps": []}
    cuda = False
    if out["torch_installed"]:
        try:
            t = torch()
            out["torch_cuda_build"] = t.version.cuda
            cuda = bool(t.cuda.is_available())
            if cuda:
                out["gpu_name"] = t.cuda.get_device_name(0)
        except Exception as e:
            out["detail"] = f"PyTorch failed to start: {e}"
    if cuda:
        out.update(ok=True, code="ok", title="GPU ready", detail=out["gpu_name"] or "")
    elif not out["torch_installed"]:
        out.update(code="no_torch", title="PyTorch isn't installed",
                   detail="Training needs PyTorch. This copy of AOMe doesn't include it.",
                   steps=["Reinstall AOMe from the full installer, or run: pip install torch --index-url https://download.pytorch.org/whl/cu124",
                          "Then restart AOMe and press Retry."])
    elif out["torch_cuda_build"] is None:
        out.update(code="cpu_torch", title="This copy of PyTorch can't use a GPU",
                   detail="The installed PyTorch is the CPU-only build, so training would run on the CPU and be very slow.",
                   steps=["Install the GPU build: pip uninstall torch, then pip install torch --index-url https://download.pytorch.org/whl/cu124",
                          "Restart AOMe and press Retry."])
    elif not (out["driver_found"] or out["nvidia_smi"]):
        out.update(code="no_nvidia", title="No NVIDIA GPU or driver found",
                   detail="AOMe can only use NVIDIA GPUs. Either this PC has no NVIDIA GPU (AMD and Intel GPUs aren't supported), or the NVIDIA driver isn't installed.",
                   steps=["If you have an NVIDIA GPU: install the latest driver from nvidia.com/drivers, restart the PC, then press Retry.",
                          "Laptops: make sure Windows uses the NVIDIA GPU for AOMe (Settings > System > Display > Graphics).",
                          "No NVIDIA GPU? You can continue on the CPU with small models (under 1B), but training is much slower."])
    else:
        out.update(code="cuda_unavailable", title="The GPU was found but can't be used",
                   detail="The NVIDIA driver is present but PyTorch can't start CUDA. The driver is usually too old for this PyTorch build, or another program holds the GPU.",
                   steps=["Update the NVIDIA driver, restart the PC, then press Retry.",
                          "Close games, video editors and other GPU-heavy programs."])
    return out
