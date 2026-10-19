# Phase 1.1: Understand the machine you will use

## Why this lesson comes first

Model size, batch size, context length, and training time all depend on the
machine available. Take an inventory before choosing a model or installing a
GPU-specific framework build. This lesson uses only Python's standard library;
it does not install packages or change system settings.

## Learning objectives

By the end, you should be able to:

- identify the operating system, Python interpreter, CPU core count, physical
  memory, and free disk space available to the project;
- distinguish a physical display adapter from an accelerator that PyTorch can
  actually use;
- describe the difference between CPU PyTorch and CUDA-enabled PyTorch; and
- explain the device policy used for the later exercises.

## Run the inventory

From the repository root, run:

```powershell
python scripts\system_info.py
```

For structured output, print JSON or save a private local copy:

```powershell
python scripts\system_info.py --json
python scripts\system_info.py --json > system-report.json
```

The generated `system-report.json` is excluded from Git because it includes
machine-specific details.

The script reports the Python executable selected by that `python` command.
This is useful for spotting when a terminal is using a different Python from
the one expected in an editor or virtual environment. Do not assume the
terminal's interpreter is the editor's selected interpreter; compare them
explicitly when working in VS Code.

The report includes CPU and memory information, free disk space for the current
working directory, and optional PyTorch runtime capability checks. It never
installs packages or writes a report file.

## Inventory the hardware separately

The script reports devices PyTorch can use, not a complete list of physical
graphics hardware. Check the operating system as well:

- **Windows:** Open Task Manager → Performance for CPU, Memory, GPU, and disk.
  To list display adapters in PowerShell, run:

  ```powershell
  Get-CimInstance Win32_VideoController | Select-Object Name
  ```

- **macOS:** Use About This Mac → System Report for chip, memory, and graphics
  information.
- **Linux:** Use the system monitor and distribution-provided hardware tools;
  `lscpu`, `free -h`, `df -h`, and (for NVIDIA hardware) `nvidia-smi` are common
  inspection commands when installed.

Record the adapter names and distinguish integrated graphics, virtual display
adapters, and compute-capable GPUs. An adapter appearing in the operating
system does not guarantee that the installed PyTorch build supports it.

## Understand the device result

PyTorch does not automatically move a model or tensors to a GPU. Code chooses
the device and moves both model and data there. For this project, the initial
recommendation policy is:

1. Use `cuda:0` if the installed PyTorch build can access CUDA.
2. Otherwise use `mps` if the PyTorch MPS backend is available (Apple
   platforms).
3. Otherwise use `cpu`.

This policy is a teaching default, not a claim that CPU is always faster or
that every GPU backend is supported. The report's "recommended device" is based
on runtime checks. CUDA availability means the current PyTorch installation
can access a CUDA device; it is not inferred merely from an adapter name.

For example, a CPU-only PyTorch build reports no CUDA build and recommends
`cpu`, even if the machine has a GPU listed by the operating system. Selecting
CUDA later requires a compatible PyTorch build, supported NVIDIA hardware, a
working driver, and compatible runtime components. Do not install or change
those components as part of this lesson; Phase 1.4 covers package installation.

## Hands-on exercise

1. Run the text report and the JSON report from the repository root.
2. Find the exact Python executable and compare it with the interpreter
   selected in VS Code.
3. Use the operating system's hardware view to note CPU, memory, graphics
   adapters, and disk space. Mark any virtual or integrated display adapters.
4. If PyTorch is already installed, compare the report's CUDA build,
   `cuda_available`, CUDA device names, and recommended device. If it is not
   installed, explain why the PyTorch device is not yet known.
5. Write a short explanation of why the project will initially use the
   reported device and what would need to be true to use an accelerator.
6. Consider whether free disk space can hold the dataset, checkpoints, and
   environment. Keep large artifacts out of Git.

## Notes template

Copy this into a private note or `docs/hardware-notes.md` and fill it with
your own machine's values. Avoid committing usernames, full local paths, or
other machine-specific details to a public repository.

```text
Date:
Operating system and version:
Python version:
Python executable matches the intended environment?:
CPU model / logical cores:
Installed memory / currently available memory:
Graphics adapters reported by the OS:
PyTorch installed? / version:
PyTorch CUDA build / CUDA available / visible CUDA devices:
PyTorch MPS available:
Recommended device and reason:
Free disk on the project volume:
Expected project constraints:
```

## Completion checklist

- [ ] I can run the inventory script from the repository root.
- [ ] I can identify which Python executable the command used.
- [ ] I can distinguish OS-listed graphics hardware from PyTorch-visible
  accelerators.
- [ ] I can explain the reported device recommendation.
- [ ] I have recorded the machine constraints without committing sensitive
  local details.

## Troubleshooting

- **`python` is not recognized:** install Python or use the command configured
  for the current environment; do not continue until you know which
  interpreter will run the project.
- **The executable is unexpected:** activate the intended environment and
  rerun the command. In VS Code, separately inspect/select the workspace
  interpreter.
- **Memory says unavailable:** the script intentionally reports unknown on
  operating systems whose memory interface is not implemented. Record it from
  the OS system monitor instead.
- **A GPU is listed but CUDA is unavailable:** the adapter may not support
  CUDA, the installed PyTorch build may be CPU-only, or driver/runtime
  compatibility may be missing. Record this for Phase 1.3 rather than
  changing the system during this lesson.
- **The script fails while importing PyTorch:** keep the full error. An
  installed package that cannot import is an environment issue to diagnose,
  not a reason to silently assume CPU support.

## Deliverables

- A completed machine-notes template.
- A successful terminal report.
- A short paragraph explaining the selected/recommended training device.

No dependency manifest or environment setup is required yet; those are Phase
1.2 and Phase 1.3.
