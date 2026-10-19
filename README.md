# Learn Transformers by Building One

This repository is a hands-on learning project for understanding how
transformer language models work and how to build the surrounding engineering
workflow. The long-term goal is to implement, train, evaluate, and serve a
small decoder-only transformer language model in PyTorch, while learning the
practices used to develop machine-learning software reliably.

> **Project status:** This is an actively developed curriculum, not a finished
> model or production service. The detailed learning path is in
> [`description.md`](./description.md). The first machine-inventory lesson is
> implemented in [`docs/phase-1.1.md`](./docs/phase-1.1.md), with a runnable
> inventory script and tests. Later phases describe planned learning and
> implementation work; they should not be mistaken for already implemented
> features.

## What you will learn

The project connects the complete model lifecycle:

1. Inspect the development machine and understand its compute limits.
2. Prepare permitted text data and create leak-resistant data splits.
3. Build a tokenizer and next-token training examples.
4. Implement transformer components and verify their tensor behavior.
5. Train a small model, diagnose it, and save resumable checkpoints.
6. Evaluate the model with quantitative and qualitative methods.
7. Generate text with explicit decoding controls.
8. Fine-tune carefully when there is a measurable reason to do so.
9. Package and serve inference with tests, resource limits, and documentation.

The project emphasizes not only model code but also data provenance, tests,
experiment tracking, reproducibility, measurement, privacy, and honest
reporting of limitations. These habits transfer to work on larger models and
other machine-learning systems.

## Meaning of “from scratch”

For the core learning project, **from scratch** means implementing and
understanding the tokenizer/model pipeline and training loop rather than
starting from pretrained weights. You may use established libraries for
framework primitives, data loading, tokenization, testing, logging, and
serving. The aim is not to reimplement every tensor operation or write a
production-grade numerical framework.

Pretrained models can be used later as clearly labeled comparison points or
optional extensions; they are not the starting weights for the core model.

## Who this is for

The curriculum is intended for learners interested in deep learning, NLP, and
the practical engineering around model development. It starts with environment
and tensor basics, then progresses toward model internals and an inference
application. You do not need a GPU to begin: early lessons and deliberately
small experiments can run on CPU.

The goal is to understand how the pieces fit together and to practice a
reproducible workflow—not to train a state-of-the-art foundation model on a
personal computer.

## Start here

### 1. Read the learning path

Start with [`description.md`](./description.md). It defines the lesson
sequence, learning tasks, verification checks, phase gates, and suggested
milestones. Treat each numbered subsection as a short work session:

1. Learn the concept.
2. Implement or investigate the task.
3. Run the verification check.
4. Record what happened and what you learned.
5. Move on after the phase gate is satisfied.

Use small datasets and small models first. If a correctness check fails, debug
it before adding more features. Keep implementation changes small enough to
review and test.

### 2. Complete Phase 1.1

Phase 1.1 takes an inventory of the machine before model size or hardware
choices are made. It requires only the Python standard library; it does not
install packages or change system settings.

From the repository root, run:

```powershell
python scripts\system_info.py
```

To print machine details as JSON:

```powershell
python scripts\system_info.py --json
```

To save that JSON locally, for example for your own notes:

```powershell
python scripts\system_info.py --json > system-report.json
```

`system-report.json` is ignored by Git because it contains machine-specific
details. Review it before sharing it anywhere.

The report includes:

- operating system and machine architecture;
- Python version, implementation, and executable path;
- CPU model when available and logical CPU count;
- total and available physical memory on supported systems;
- free disk space for the current working directory; and
- optional PyTorch version and runtime availability for CUDA and Apple MPS.

The script reports accelerators PyTorch can currently use; it is not a
complete inventory of physical graphics adapters. An adapter listed by the
operating system does not necessarily mean that the installed PyTorch build
supports it. See [`docs/phase-1.1.md`](./docs/phase-1.1.md) for OS-specific
hardware checks, the device-selection policy, a note-taking template,
troubleshooting guidance, and completion checklist.

The current script recommends CUDA when PyTorch reports CUDA as available,
otherwise MPS when available, and otherwise CPU. This is a teaching default;
it is not a benchmark or a claim that a GPU is always faster.

### 3. Run the Phase 1.1 tests

The tests use Python's built-in `unittest` runner and need no extra test
package:

```powershell
python -m unittest discover -s tests -v
```

The tests cover byte formatting for memory and the CUDA/MPS/CPU device
recommendation policy.

## Learning roadmap

The complete curriculum currently has nine phases and 69 lessons. Consult
[`description.md`](./description.md) for every lesson's specific
learn/do/verify tasks and each phase's completion gate.

| Phase | Focus | Practical outcome |
| --- | --- | --- |
| 1. Environment setup | Machine inventory, repository and environment setup, dependencies, tensors, and quality checks | A reproducible development setup |
| 2. Data preparation | Task definition, data audit, cleaning, split design, tokenization, examples, and loading | Reproducible, inspectable training data |
| 3. Model architecture | Embeddings, attention, masking, transformer blocks, model configuration, and tests | A small causal language model with tested behavior |
| 4. Training | Loss, baselines, training loop, validation, checkpoints, metrics, and debugging | A measured, resumable training run |
| 5. Evaluation | Held-out metrics, perplexity, fixed prompts, comparisons, and limitations | An honest and repeatable evaluation |
| 6. Text generation | Autoregressive loop, stopping rules, decoding, efficiency, and API tests | A tested generation interface |
| 7. Fine-tuning | Task suitability, data preparation, conservative adaptation, and comparison | An evidence-based keep/reject decision |
| 8. Deployment | Model packaging, API, limits, performance, observability, and release | A small, documented inference application |
| 9. Continued learning | Explain and reproduce the project, work-style change, experiments, and specialization | A maintainable, portfolio-ready learning project |

Fine-tuning is intentionally placed after training and evaluation. A small
model trained on a limited corpus may not have enough capability for
fine-tuning to be useful; the curriculum asks you to establish a measurable
reason and compare against an unchanged base model before keeping an adapted
checkpoint.

## Repository contents

### Present in the repository

```text
.
├── .gitignore
├── LICENSE
├── README.md
├── description.md
├── docs/
│   └── phase-1.1.md
├── scripts/
│   └── system_info.py
└── tests/
    └── test_system_info.py
```

- **`README.md`** — project overview, status, getting started, and contribution
  to your own learning workflow.
- **`description.md`** — the detailed curriculum: lessons, exercises,
  verification checks, phase gates, references, and milestones.
- **`docs/phase-1.1.md`** — detailed first lesson on machine resources and
  runtime device availability.
- **`scripts/system_info.py`** — dependency-free system inventory tool with
  optional PyTorch capability reporting.
- **`tests/test_system_info.py`** — focused tests for inventory utility
  behavior.
- **`.gitignore`** — ignores Python environments and common generated files,
  plus the local `system-report.json`.
- **`LICENSE`** — the license file included in this repository. Review it
  before redistributing the project or its contents.

### Planned as the project grows

The curriculum suggests adding items such as `configs/`, `data/`, `notebooks/`,
`src/`, `artifacts/`, and additional documentation and tests as they become
necessary. These directories and the model implementation are not all present
yet. In particular, this repository does not currently provide a tokenizer,
transformer model, training loop, trained checkpoint, text-generation
interface, or inference API.

Add structure when a lesson needs it rather than creating empty scaffolding in
advance. Keep source code separate from downloaded data and generated
artifacts.

## Development environment

Phase 1.1 is intentionally dependency-free apart from optional PyTorch
inspection. The inventory script imports PyTorch only if it is installed. If
PyTorch is absent, it still reports general system information and explains
that framework device capability cannot yet be checked.

No project-wide dependency manifest has been added yet. Environment creation
and selecting a compatible PyTorch build are covered in later Phase 1 lessons.
Do not assume that the machine inventory alone proves a CUDA setup is ready:
hardware, drivers, framework build, and runtime compatibility all matter.

### Windows

Run commands from the repository root in PowerShell:

```powershell
python --version
python scripts\system_info.py
python -m unittest discover -s tests -v
```

To inspect display adapters separately:

```powershell
Get-CimInstance Win32_VideoController | Select-Object Name
```

### macOS and Linux

Use the platform's Python command and a path separator appropriate for that
shell:

```sh
python3 --version
python3 scripts/system_info.py
python3 -m unittest discover -s tests -v
```

On macOS and Linux, the script reports physical memory as unavailable if its
current platform-specific memory reader does not support that operating
system. In that case, obtain memory details from the operating system's system
monitor. The system inventory is a learning aid, not a replacement for a
hardware diagnostic tool.

## How to learn effectively

### Build correctness before scale

Start with tiny, inspectable inputs and configurations. For example, manually
check a short token sequence and its shifted next-token targets before
processing a large corpus. Verify tensor shapes and causal masking before a
long training run. Scale only after the basic behavior is correct.

### Keep experiments reproducible

As later phases add training, connect results to the code revision,
configuration, dataset split, tokenizer, and checkpoint that produced them.
Set random seeds where practical, while recognizing that hardware and
computational kernels may still introduce nondeterminism.

### Make one controlled change at a time

Write a hypothesis before changing model width, context length, optimizer, or
decoding settings. Preserve a baseline and compare runs on the same evaluation
procedure. Record failed experiments and what they ruled out.

### Treat tests as part of learning

Test expected behavior and failure cases. Useful model checks include tensor
shapes, finite outputs and gradients, causal behavior, data alignment, loss
masking, checkpoint reload, and generation boundary cases. When a check fails,
find the source of the error before adding new functionality.

### Handle data and generated artifacts responsibly

Use data you are allowed to process. Record source, license, and relevant
restrictions. Keep private or sensitive data, prompts, machine reports, logs,
and large checkpoints out of public repositories unless there is a clear,
reviewed reason and permission to share them.

## Milestones

1. **Environment:** Complete the inventory, then establish a clean setup and
   repeatable quality checks.
2. **Data:** Produce documented, leak-resistant splits, a tokenizer, and
   verified input-target batches.
3. **Model:** Implement a causal transformer with documented shapes and tests.
4. **Training:** Demonstrate a tiny-batch overfit check, compare against a
   baseline, and save a resumable checkpoint.
5. **Evaluation and generation:** Create a fixed evaluation suite, report
   metrics with limitations, and test generation behavior.
6. **Application:** Package a loadable model bundle and a bounded local
   inference API.
7. **Portfolio:** Reproduce the workflow from a clean setup and document
   results, data provenance, model limitations, and a reviewed work-style
   change.

These are completion milestones, not claims about the current repository
status. Use the phase gates in [`description.md`](./description.md) to decide
when each one is actually complete.

## Responsible use and limitations

The model eventually trained through this curriculum is an educational
artifact. A small model trained from scratch on a small corpus can produce
incoherent, biased, repetitive, or unsupported text. Training loss or
perplexity alone does not establish usefulness, factuality, fairness, or
safety.

Before using or sharing data, check its source, terms, privacy implications,
and suitability. Before presenting model outputs, evaluate representative
successes and failures and state what the model should not be trusted to do.
For any future service, validate and bound inputs, avoid logging prompt text by
default, load only trusted model artifacts, and document the service's intended
use and limitations.

## References

The curriculum links additional references at the relevant phase. Useful
starting points include:

- [Python virtual environments](https://docs.python.org/3/library/venv.html)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [PyTorch tensor basics](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [PyTorch data loading](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Contributing to your own learning

Treat each improvement like a small work task:

1. State the problem and acceptance criteria.
2. Make a focused change on a branch.
3. Add or update tests and documentation.
4. Run the relevant checks and inspect the diff.
5. Summarize the result, trade-offs, and anything that remains uncertain.

The project is designed to grow incrementally. Prefer understandable,
reproducible improvements over complexity added for its own sake.
