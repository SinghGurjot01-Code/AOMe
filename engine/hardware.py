"""Live hardware readings: CPU, RAM, GPU, VRAM, disk, and this process."""
import os
import platform
import shutil
import threading
import time

import psutil

from .config import HOME

try:
    import pynvml
    pynvml.nvmlInit()
    _NV = True
except Exception:
    pynvml = None
    _NV = False

def reset_nvml():
    """Try to (re)connect to the NVIDIA driver, e.g. after the user installs or updates it."""
    global pynvml, _NV
    try:
        import pynvml as _p
        _p.nvmlInit()
        pynvml, _NV = _p, True
    except Exception:
        _NV = False
    return _NV


_proc = psutil.Process(os.getpid())
_proc.cpu_percent(None)
psutil.cpu_percent(None)
_cpu_name = None


def cpu_name() -> str:
    global _cpu_name
    if _cpu_name:
        return _cpu_name
    name = ""
    try:
        if platform.system() == "Windows":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        elif platform.system() == "Darwin":
            import subprocess
            name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        pass
    _cpu_name = name or platform.processor() or "CPU"
    return _cpu_name


def _gpu():
    """Return the primary NVIDIA GPU stats, or None. Falls back to torch for name/VRAM only."""
    if _NV:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                power = None
            return {"name": name, "vram_total_mb": mem.total // 1048576, "vram_free_mb": mem.free // 1048576,
                    "util": util.gpu, "temp_c": temp, "power_w": power}
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return {"name": torch.cuda.get_device_name(0), "vram_total_mb": total // 1048576,
                    "vram_free_mb": free // 1048576, "util": None, "temp_c": None, "power_w": None}
    except Exception:
        pass
    return None


def snapshot() -> dict:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(str(HOME))
    g = _gpu()
    try:
        proc_rss = _proc.memory_info().rss // 1048576
        proc_cpu = _proc.cpu_percent(None) / max(1, psutil.cpu_count() or 1)
    except Exception:
        proc_rss, proc_cpu = 0, 0.0
    return {
        "device_name": g["name"] if g else cpu_name(),
        "cuda_available": bool(g),
        "cpu_name": cpu_name(),
        "cpu_cores": psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        "cpu_threads": psutil.cpu_count() or 1,
        "cpu_percent": round(psutil.cpu_percent(None), 1),
        "ram_total_mb": vm.total // 1048576,
        "ram_free_mb": vm.available // 1048576,
        "ram_percent": round(vm.percent, 1),
        "gpu_name": g["name"] if g else None,
        "gpu_percent": g["util"] if g else None,
        "gpu_temp_c": g["temp_c"] if g else None,
        "gpu_power_w": g["power_w"] if g else None,
        "vram_total_mb": g["vram_total_mb"] if g else 0,
        "vram_free_mb": g["vram_free_mb"] if g else 0,
        "disk_total_gb": round(disk.total / 1073741824, 1),
        "disk_free_gb": round(disk.free / 1073741824, 1),
        "engine_ram_mb": proc_rss,
        "engine_cpu_percent": round(proc_cpu, 1),
        "os": f"{platform.system()} {platform.release()}",
        "ts": time.time(),
    }
