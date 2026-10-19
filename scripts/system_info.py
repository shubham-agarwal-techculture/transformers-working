"""Report the machine and Python runtime available for transformer experiments."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

GIB = 1024 ** 3


def _windows_memory() -> Tuple[int, int]:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _linux_memory() -> Tuple[int, int]:
    memory_values: Dict[str, int] = {}
    with open("/proc/meminfo", encoding="ascii") as memory_file:
        for line in memory_file:
            key, value = line.split(":", maxsplit=1)
            if key in {"MemTotal", "MemAvailable"}:
                memory_values[key] = int(value.strip().split()[0]) * 1024

    return memory_values["MemTotal"], memory_values["MemAvailable"]


def get_memory_info() -> Dict[str, Optional[int]]:
    """Return total and available physical memory in bytes when supported."""
    if sys.platform == "win32":
        total_bytes, available_bytes = _windows_memory()
    elif sys.platform.startswith("linux"):
        total_bytes, available_bytes = _linux_memory()
    else:
        total_bytes = None
        available_bytes = None

    return {
        "total_bytes": total_bytes,
        "available_bytes": available_bytes,
    }


def recommend_torch_device(cuda_available: bool, mps_available: bool) -> str:
    """Choose the first available device under this project's documented policy."""
    if cuda_available:
        return "cuda:0"
    if mps_available:
        return "mps"
    return "cpu"


def get_torch_info() -> Dict[str, Any]:
    """Inspect PyTorch if installed; never install or configure it."""
    if importlib.util.find_spec("torch") is None:
        return {
            "installed": False,
            "recommended_device": None,
        }

    import torch

    cuda_available = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    cuda_devices = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if cuda_available
        else []
    )
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_devices": cuda_devices,
        "mps_available": mps_available,
        "recommended_device": recommend_torch_device(
            cuda_available=cuda_available,
            mps_available=mps_available,
        ),
    }


def collect_system_info() -> Dict[str, Any]:
    """Collect a local machine inventory without requiring third-party packages."""
    disk = shutil.disk_usage(Path.cwd())
    return {
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "cpu": {
            "processor": platform.processor() or None,
            "logical_cores": os.cpu_count(),
        },
        "memory": get_memory_info(),
        "disk": {
            "checked_path": str(Path.cwd()),
            "free_bytes": disk.free,
            "total_bytes": disk.total,
        },
        "pytorch": get_torch_info(),
    }


def format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unavailable"
    return "{:.2f} GiB".format(value / GIB)


def print_report(report: Dict[str, Any]) -> None:
    operating_system = report["operating_system"]
    python = report["python"]
    cpu = report["cpu"]
    memory = report["memory"]
    disk = report["disk"]
    torch_info = report["pytorch"]

    print("System inventory")
    print("  OS: {} ({}, {})".format(
        operating_system["platform"],
        operating_system["system"],
        operating_system["architecture"],
    ))
    print("  Python: {} ({})".format(python["version"], python["implementation"]))
    print("  Python executable: {}".format(python["executable"]))
    print("  CPU: {}".format(cpu["processor"] or "model name unavailable"))
    print("  Logical CPU cores: {}".format(cpu["logical_cores"] or "unavailable"))
    print("  Physical memory: {}".format(format_bytes(memory["total_bytes"])))
    print("  Available memory: {}".format(format_bytes(memory["available_bytes"])))
    print("  Free disk at {}: {}".format(
        disk["checked_path"],
        format_bytes(disk["free_bytes"]),
    ))

    if not torch_info["installed"]:
        print("  PyTorch: not installed (device capability will be checked later)")
        return

    print("  PyTorch: {}".format(torch_info["version"]))
    print("  CUDA build: {}".format(torch_info["cuda_build"] or "none"))
    print("  CUDA available: {}".format(torch_info["cuda_available"]))
    print("  CUDA devices: {}".format(
        ", ".join(torch_info["cuda_devices"]) or "none available to PyTorch"
    ))
    print("  MPS available: {}".format(torch_info["mps_available"]))
    print("  Recommended device (CUDA, then MPS, then CPU): {}".format(
        torch_info["recommended_device"]
    ))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report local system resources for machine-learning experiments."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the report as JSON instead of human-readable text",
    )
    arguments = parser.parse_args()

    report = collect_system_info()
    if arguments.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
