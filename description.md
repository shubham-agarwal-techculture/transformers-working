# Learn transformers by building one

This curriculum is designed for **learning by doing**, from first principles
through a small, usable inference service. You will build a decoder-only
transformer language model in PyTorch, understand each part, train it on a
permitted dataset, evaluate it, and package it for use. The same habits—
testing, reproducibility, measurement, documentation, and careful data
handling—are useful in professional machine-learning work.

Here, **from scratch** means implementing the tokenizer/model pipeline and
training loop yourself for learning. You may use established libraries for
data loading, tokenization, logging, testing, and serving. Do not use pretrained
weights for the core model; use pretrained models later only for comparison or
as a clearly labeled optional extension.

## How to use this curriculum

- Treat each numbered lesson as a small work session. Read the concept, make
  the change, run the check, and write down what you learned before moving on.
- Keep one repository throughout. Prefer small, reviewable changes over a
  single large notebook or script. Use notebooks for exploration, then move
  reusable logic into tested Python modules.
- Start with tiny data and a tiny model so experiments finish quickly. Scale
  only after correctness is demonstrated; the goal is to learn the engineering
  process, not to train a competitive foundation model on limited hardware.
- Record questions, failed experiments, decisions, and results. A failed run
  with a useful diagnosis is a successful learning exercise.
- At each phase gate, complete the stated checks before building on the work.
  If a check fails, debug it rather than layering more features on top.
- Use only data and examples you are permitted to process. Do not put private
  or sensitive data into experiments, logs, prompts, or public repositories.

## Suggested project layout

Adapt this as the project grows; do not create every file on day one.

```text
transformer-from-scratch/
├── README.md
├── pyproject.toml                 # or requirements.txt
├── configs/
│   └── tiny.yaml
├── data/                          # normally ignored by Git
├── docs/
│   ├── decisions.md
│   ├── experiments.md
│   └── model-card.md
├── notebooks/                     # exploration, not production entry points
├── scripts/
├── src/
│   └── transformer_lm/
├── tests/
└── artifacts/                     # checkpoints, normally ignored by Git
```

## Phase 1: Set up a professional development environment

**Outcome:** A clean, reproducible project that runs a small PyTorch program.

### 1.1 Understand the machine you will use
- **Learn:** Python version, CPU/GPU, device memory, disk space, and the
  difference between a CPU build and a CUDA-enabled PyTorch build.
- **Do:** Record the operating system, Python version, available accelerator,
  and memory. Check the compute limits before selecting a model size.
- **Verify:** Explain which device PyTorch will use and why.

### 1.2 Create the repository
- **Learn:** Why source code, raw data, generated artifacts, and documentation
  should not all be mixed together.
- **Do:** Initialize a Git repository, create a README, and add `.gitignore`
  entries for virtual environments, downloaded data, logs, and checkpoints.
- **Verify:** `git status` is clean after adding only the intended starter
  files; no data or model files are accidentally tracked.

### 1.3 Isolate the Python environment
- **Learn:** Virtual environments, dependency manifests, and why globally
  installed packages make a project difficult to reproduce.
- **Do:** Create and activate a virtual environment. Add a minimal
  `pyproject.toml` or `requirements.txt`.
- **Verify:** A new terminal can activate the environment and report the
  expected interpreter.

### 1.4 Install only the initial dependencies
- **Learn:** Runtime dependencies versus development dependencies.
- **Do:** Install PyTorch using its official selector, then add only the
  packages needed for the next lessons (for example, NumPy and pytest).
- **Verify:** Save the installed package versions and run `pip check` or the
  package manager's equivalent.

### 1.5 Exercise tensors and devices
- **Learn:** Tensor shape, dtype, device, broadcasting, and gradient tracking.
- **Do:** Write a short script that creates tensors, performs arithmetic, and
  computes a simple gradient.
- **Verify:** Assert expected shapes and values. If a GPU is available, move a
  tensor to it and confirm the result is on that device.

### 1.6 Add the first repeatable quality checks
- **Learn:** The roles of formatting, linting, unit tests, and version control.
- **Do:** Add a small test that runs the tensor exercise; choose a formatter
  and linter only if useful for the project.
- **Verify:** Run the test from the command line and record the exact command
  in the README.

**Phase gate:** Another person can clone the repository, create the
environment, run the tensor script, and run its test using your instructions.

**Keep:** setup commands, dependency versions, hardware notes, first Git
history, and a short note explaining any environment issue you solved.

**References:** [Python `venv`](https://docs.python.org/3/library/venv.html),
[PyTorch installation](https://pytorch.org/get-started/locally/),
[PyTorch tensor tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html),
[pytest getting started](https://docs.pytest.org/en/stable/getting-started.html).

## Phase 2: Explore and prepare a text dataset

**Outcome:** Versioned, documented data-preparation code and leak-resistant
training, validation, and test splits.

### 2.1 Define a useful learning task
- **Learn:** The difference between a dataset, a task, and an evaluation goal.
- **Do:** Write a one-paragraph task brief: what text will be continued, who
  might use the result, what success could look like, and what is out of scope.
- **Verify:** Define at least one measurable signal and one qualitative
  criterion before training.

### 2.2 Select a small, permitted corpus
- **Learn:** Data provenance, licenses, terms of use, privacy, and
  representativeness.
- **Do:** Choose a small public or self-created corpus suitable for a first
  run. Record its source, license, download date, and intended use.
- **Verify:** Confirm that the data can legally and ethically be used for this
  experiment. Exclude personal or sensitive data.

### 2.3 Inspect the raw format
- **Learn:** How file encodings, schemas, missing values, and malformed rows
  affect a pipeline.
- **Do:** Write a loader that reports file count, row/document count, fields,
  character counts, and a few representative examples.
- **Verify:** Fail clearly on unexpected encodings or missing required fields;
  do not silently discard bad records.

### 2.4 Create a data audit
- **Learn:** Duplicates, empty records, outliers, language mix, and source
  leakage.
- **Do:** Produce a small audit report with counts, duplicate rates, length
  percentiles, and examples requiring manual review.
- **Verify:** Check that the report is reproducible and does not expose
  sensitive text.

### 2.5 Implement explicit cleaning rules
- **Learn:** Normalization is a modeling decision; removing punctuation,
  casing, markup, or whitespace can change what the model learns.
- **Do:** Write named, testable cleaning functions and apply only justified
  transformations.
- **Verify:** Add tests for empty input, whitespace, punctuation, Unicode, and
  a normal example. Inspect before-and-after samples.

### 2.6 Split the data without leakage
- **Learn:** Why random rows can leak near-duplicate passages across splits.
- **Do:** Split by document, author, source, or time when appropriate. Save
  split IDs and the random seed.
- **Verify:** Assert that split identifiers are disjoint and report counts and
  proportions. Do not use the test split to make training decisions.

### 2.7 Learn tokenization with a baseline
- **Learn:** Character, word, and subword tokenization; vocabulary size;
  unknown tokens; and special tokens.
- **Do:** First build a tiny character tokenizer to understand encode/decode.
  Then train or select a subword tokenizer using training data only.
- **Verify:** Test round trips on ordinary text, punctuation, Unicode, empty
  text, and text containing unseen characters or tokens. Report vocabulary
  size and token counts.

### 2.8 Create next-token examples
- **Learn:** Context windows, shifted targets, padding, truncation, and
  document boundaries.
- **Do:** For a short token sequence, print the input and its one-token-shifted
  target by hand. Then implement batching and the chosen boundary policy.
- **Verify:** Assert that every input-target pair is aligned correctly, padding
  is masked, and examples do not accidentally cross document boundaries.

### 2.9 Build a reliable data loader
- **Learn:** Dataset and DataLoader responsibilities, shuffling, batching, and
  worker processes.
- **Do:** Implement a small dataset class or equivalent loader. Make shuffling
  apply to training only.
- **Verify:** Inspect batch shapes, dtypes, masks, and decoded examples. Run
  the same seeded setup twice and compare its first batch.

**Phase gate:** A single documented command produces inspectable splits,
tokenizer files, and valid batches. Tests catch a deliberately broken
input-target alignment.

**Keep:** task brief, source/license notes, audit report, cleaning tests, split
metadata, tokenizer settings, and representative encoded examples.

**References:** [Hugging Face Datasets](https://huggingface.co/docs/datasets/),
[Tokenizers](https://huggingface.co/docs/tokenizers/),
[PyTorch data loading](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

## Phase 3: Build the model from understandable components

**Outcome:** A small causal language model with tested tensor shapes and
verified causal behavior.

### 3.1 Write the model contract before coding
- **Learn:** Inputs, outputs, shape conventions, configuration, and interface
  contracts.
- **Do:** Specify input IDs, optional attention mask, output logits, tensor
  shapes, vocabulary size, and context limit.
- **Verify:** Write shape examples for batch size one and greater than one;
  define behavior for invalid token IDs and overlong sequences.

### 3.2 Implement and test embeddings
- **Learn:** Embedding lookup as a mapping from token IDs to vectors.
- **Do:** Implement token embeddings and positional embeddings as separate
  modules.
- **Verify:** Check output shape, dtype, parameter count, and that token IDs
  outside the vocabulary fail with a useful error.

### 3.3 Understand attention with a tiny example
- **Learn:** Queries, keys, values, dot products, scaling, softmax, and weighted
  sums.
- **Do:** Calculate attention by hand for a tiny matrix, then reproduce it with
  PyTorch tensor operations.
- **Verify:** Compare the hand calculation and implementation within a
  numerical tolerance.

### 3.4 Implement causal masking
- **Learn:** Why a language model must not see future tokens when predicting
  the next one.
- **Do:** Build a causal mask and apply it to attention scores before softmax.
- **Verify:** Test the mask values and prove that changing future input tokens
  cannot change earlier-position outputs.

### 3.5 Implement one attention head
- **Learn:** Learned projections and the shapes of query, key, value, and
  attention output tensors.
- **Do:** Implement a single-head self-attention module with explicit shape
  checks.
- **Verify:** Test batch size one, multiple batches, short sequence, and
  maximum context length. Confirm all outputs are finite.

### 3.6 Extend attention to multiple heads
- **Learn:** How heads divide and combine representation dimensions.
- **Do:** Split projections into heads, compute attention per head, then join
  them and apply an output projection.
- **Verify:** Assert that the embedding dimension is divisible by the number
  of heads, and test the complete shape transformation.

### 3.7 Build the feed-forward sublayer
- **Learn:** The position-wise MLP, expansion dimension, activation, and
  dropout.
- **Do:** Implement the MLP and test that it preserves the expected sequence
  and batch dimensions.
- **Verify:** Confirm the parameter count and test train/evaluation behavior
  when dropout is enabled.

### 3.8 Assemble a transformer block
- **Learn:** Residual connections, normalization, sublayer ordering, and
  gradient flow.
- **Do:** Combine attention and feed-forward sublayers with residual paths and
  normalization.
- **Verify:** Run a forward pass and backward pass; check that expected
  parameters receive finite gradients.

### 3.9 Stack blocks and add the language-model head
- **Learn:** Repeated blocks, final normalization, and mapping hidden states
  back to vocabulary logits.
- **Do:** Build a configurable stack and a final projection. Consider weight
  tying as an explicit optional experiment, not an implicit assumption.
- **Verify:** Assert logits have shape `[batch, sequence, vocabulary]` and
  calculate the total parameter count.

### 3.10 Add configuration and validation
- **Learn:** Why model settings should be explicit, serializable, and validated.
- **Do:** Create a configuration object for vocabulary size, context length,
  dimensions, heads, layers, dropout, and other selected settings.
- **Verify:** Reject invalid configurations with clear messages. Recreate the
  same model from a saved configuration.

### 3.11 Test correctness, not just shape
- **Learn:** Unit tests, invariants, numerical checks, and failure-oriented
  testing.
- **Do:** Add tests for causal behavior, valid/invalid shapes, gradients,
  determinism in evaluation mode, and parameter count.
- **Verify:** Deliberately break a mask or shape and confirm that a test fails
  for the expected reason.

**Phase gate:** The model passes tests on CPU, produces finite logits and
gradients, respects causal masking, and fits the planned tiny experiment.

**Keep:** model contract, architecture sketch, configuration, shape notes,
parameter count, tests, and decisions about architecture variants.

**References:** [PyTorch `nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html),
[Scaled dot-product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).

## Phase 4: Train, debug, and reproduce experiments

**Outcome:** A repeatable training run with trustworthy loss reporting and
checkpoints that can be resumed.

### 4.1 Calculate next-token loss by hand
- **Learn:** Logits, target IDs, cross-entropy, reduction, and ignored labels.
- **Do:** Compute loss for a tiny example and compare the result with PyTorch
  cross-entropy.
- **Verify:** Confirm the target alignment and ensure masked/padded positions
  do not contribute to the loss.

### 4.2 Establish a baseline
- **Learn:** Why simple baselines make complex systems easier to debug.
- **Do:** Measure a uniform/random predictor or a tiny character-level model
  before training the transformer.
- **Verify:** Record the baseline loss and use it to interpret whether the
  transformer learns anything.

### 4.3 Make a minimal training step
- **Learn:** Forward pass, loss, gradient clearing, backpropagation, and
  optimizer update.
- **Do:** Train one batch and report loss before and after the update.
- **Verify:** Check that gradients are present and finite and that at least
  one intended parameter changes.

### 4.4 Overfit a tiny batch deliberately
- **Learn:** How intentional overfitting isolates bugs in data, masking, loss,
  and optimization.
- **Do:** Train on one small batch repeatedly with a tiny model.
- **Verify:** The loss should decrease substantially. If not, inspect targets,
  gradients, learning rate, and causal masking before proceeding.

### 4.5 Add a training loop
- **Learn:** Epochs/steps, batch iteration, optimizer lifecycle, and device
  placement.
- **Do:** Implement a loop with explicit train mode, gradient reset, forward
  pass, loss, backward pass, and update.
- **Verify:** Run a short CPU job and confirm step count, loss, and elapsed
  time are reported correctly.

### 4.6 Add validation without training side effects
- **Learn:** The distinction between training and evaluation modes and why
  validation must not update weights.
- **Do:** Evaluate under inference mode at a configurable interval.
- **Verify:** Compare model weights before and after validation; confirm they
  are unchanged and dropout is disabled.

### 4.7 Configure experiments explicitly
- **Learn:** Hyperparameters, random seeds, configuration files, and
  experiment tracking.
- **Do:** Move settings out of code into a small configuration file. Give each
  run an ID and record its source revision and data/tokenizer identity.
- **Verify:** Re-run the same small configuration and compare expected
  behavior; note any nondeterministic hardware operations.

### 4.8 Measure training health and performance
- **Learn:** Loss curves, throughput, memory use, gradient norms, and
  overfitting.
- **Do:** Log training/validation loss, learning rate, step time, tokens per
  second, and peak memory when available.
- **Verify:** Review a plot and identify whether the run is learning,
  overfitting, unstable, or compute-bound.

### 4.9 Save and resume a checkpoint
- **Learn:** Model state, optimizer state, training step, configuration, and
  random state.
- **Do:** Save periodic checkpoints and a best-validation checkpoint. Implement
  resume logic.
- **Verify:** Stop a run, resume it, and confirm step/configuration continuity.
  Compare a short uninterrupted run with a save-and-resume run where feasible.

### 4.10 Handle common training constraints
- **Learn:** Gradient clipping, learning-rate schedules, mixed precision,
  gradient accumulation, and their trade-offs.
- **Do:** Add only one technique at a time when the baseline is correct; keep
  each behind a documented configuration choice.
- **Verify:** Compare speed, memory, and loss behavior against the baseline.
  Remove techniques that do not solve a real constraint.

### 4.11 Diagnose failure cases
- **Learn:** Symptoms of incorrect data, bad learning rate, exploding
  gradients, overfitting, and device errors.
- **Do:** Create an experiment log with a hypothesis, one controlled change,
  outcome, and next action.
- **Verify:** Explain each change using measured evidence rather than
  changing several settings at once.

**Phase gate:** A short run learns beyond the baseline, validation works
without changing weights, and a checkpoint can be loaded and resumed.

**Keep:** run configs, source revision, metrics, plots, checkpoint metadata,
debugging notes, and a short experiment summary.

**References:** [PyTorch optimization tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html),
[Saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html),
[Automatic mixed precision](https://pytorch.org/docs/stable/amp.html).

## Phase 5: Evaluate the model honestly

**Outcome:** Reproducible quantitative and qualitative evaluation with stated
limitations.

### 5.1 Separate model selection from final testing
- **Learn:** Training, validation, and test roles and how repeated test-set
  tuning biases results.
- **Do:** Select a checkpoint using validation results; define the final test
  procedure before running it.
- **Verify:** Record which data influenced each decision and keep test results
  out of iterative tuning.

### 5.2 Compute token-level loss correctly
- **Learn:** Per-token versus per-batch averaging and why padding changes
  aggregate metrics.
- **Do:** Aggregate total negative log-likelihood and valid token counts over
  the full split.
- **Verify:** Compare the aggregate with a hand-computed tiny example.

### 5.3 Report perplexity with context
- **Learn:** Perplexity as an exponential transform of average token
  cross-entropy and why tokenizers affect comparisons.
- **Do:** Calculate perplexity from the correctly aggregated loss.
- **Verify:** Report tokenizer, split, context length, and checkpoint alongside
  the score; do not compare scores from incompatible tokenization setups as if
  directly equivalent.

### 5.4 Build fixed qualitative tests
- **Learn:** Why a fixed prompt suite helps compare model versions.
- **Do:** Create prompts for normal use, short/empty input, long input,
  unusual punctuation, and likely failure cases.
- **Verify:** Run every checkpoint against the same prompt set and preserve
  both good and bad outputs.

### 5.5 Compare checkpoints and baselines
- **Learn:** Metric movement, uncertainty, and the limits of small test sets.
- **Do:** Compare the baseline, selected transformer, and any later fine-tuned
  version on identical held-out data.
- **Verify:** Report scores and representative outputs together; avoid
  claiming quality from a single metric.

### 5.6 Examine data quality, bias, and safety
- **Learn:** How data composition shapes model behavior and what a small
  evaluation cannot establish.
- **Do:** Review generated output for memorization, biased patterns,
  inappropriate content, and unsupported claims.
- **Verify:** Document observed failures and a plan for mitigation or scope
  limits.

**Phase gate:** A person who did not run the experiment can reproduce the
evaluation and understand what the results do and do not demonstrate.

**Keep:** evaluation script, metric definitions, test protocol, prompt suite,
results table, example outputs, and limitations.

**References:** [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/),
[PyTorch inference mode](https://pytorch.org/docs/stable/generated/torch.inference_mode.html).

## Phase 6: Generate text and improve inference

**Outcome:** A tested generation interface with understandable decoding
controls and measured inference costs.

### 6.1 Build the simplest generation loop
- **Learn:** Conditioning on a prompt, selecting the final-position logits, and
  autoregressive generation.
- **Do:** Encode a prompt, generate one next token, append it, and repeat.
- **Verify:** Print token IDs at each step and confirm decoded output matches
  the model's actual inputs.

### 6.2 Define prompt and stopping behavior
- **Learn:** Start/end tokens, context limits, truncation, and stopping
  criteria.
- **Do:** Handle empty prompts, overlong prompts, maximum new tokens, and
  end-of-sequence tokens explicitly.
- **Verify:** Test each boundary case, including a prompt exactly at the
  context limit.

### 6.3 Implement greedy decoding
- **Learn:** Argmax selection and why deterministic decoding is a useful
  correctness baseline.
- **Do:** Add greedy decoding with a fixed maximum output length.
- **Verify:** Repeated calls with the same model and prompt produce the same
  token sequence.

### 6.4 Add temperature and sampling
- **Learn:** Temperature scaling, probability distributions, top-k, and
  nucleus (top-p) sampling.
- **Do:** Implement one sampling strategy at a time with validated parameter
  ranges and a controllable random generator.
- **Verify:** A fixed seed is repeatable. Compare several settings on the same
  prompts and record differences rather than assuming more randomness means
  better text.

### 6.5 Compare optional decoding methods
- **Learn:** Beam search and why it may favor generic continuations in
  open-ended generation.
- **Do:** Add beam search only as an optional learning experiment.
- **Verify:** Compare quality, latency, and repetition against sampling on the
  same prompt suite.

### 6.6 Make generation safe and efficient
- **Learn:** Evaluation mode, inference mode, memory use, and key-value caching.
- **Do:** Disable gradients and measure latency for short and longer outputs.
  Implement a key-value cache only after uncached generation is correct.
- **Verify:** Cached and uncached generation agree under deterministic
  decoding; record speed and memory differences.

### 6.7 Package a reusable generation API
- **Learn:** Separating model loading, input validation, inference, and output
  formatting.
- **Do:** Implement a small Python function with typed inputs, defaults, and
  clear errors. Keep model code independent from a future web framework.
- **Verify:** Add tests for output type/length, invalid options, context
  overflow, stop tokens, and deterministic mode.

**Phase gate:** The generation API passes tests for normal and boundary cases,
and fixed prompts produce reproducible results when configured to do so.

**Keep:** decoding design, API contract, prompt suite outputs, latency
measurements, and caching comparisons.

**References:** [Generation strategies](https://huggingface.co/docs/transformers/generation_strategies),
[PyTorch inference mode](https://pytorch.org/docs/stable/generated/torch.inference_mode.html).

## Phase 7: Fine-tune for a focused use case

**Outcome:** A controlled adaptation experiment with a fair comparison to the
unchanged base checkpoint.

### 7.1 Decide whether fine-tuning is the right tool
- **Learn:** Fine-tuning versus better data, prompting, retrieval, and
  pretraining; understand that a tiny scratch-trained model may not have
  enough capability to adapt usefully.
- **Do:** State the task-specific gap and the evidence that fine-tuning could
  address it.
- **Verify:** Define a measurable target and a stop condition before spending
  compute.

### 7.2 Create task-specific examples
- **Learn:** Input/output formats, instruction examples, domain text, and
  leakage risks.
- **Do:** Prepare permitted examples using the existing tokenizer and model
  objective unless a change is explicitly justified.
- **Verify:** Inspect examples, deduplicate them, and keep evaluation examples
  separate by source where appropriate.

### 7.3 Establish a base-model score
- **Learn:** Why the unfine-tuned model is the required control.
- **Do:** Evaluate the base model on task and general-purpose prompts before
  training.
- **Verify:** Save scores and generated samples with checkpoint identity.

### 7.4 Run a conservative fine-tuning experiment
- **Learn:** Lower learning rates, short runs, early stopping, and overfitting.
- **Do:** Start from a copy/reference of the selected checkpoint, use a
  conservative configuration, and preserve the base artifacts.
- **Verify:** Monitor validation loss and outputs; stop if task results worsen
  or general behavior regresses materially.

### 7.5 Compare and make a keep/reject decision
- **Learn:** Regression testing and multi-objective trade-offs.
- **Do:** Run the same task and general-purpose evaluation suite on base and
  adapted checkpoints.
- **Verify:** Keep the adaptation only when it meets the pre-stated target
  without unacceptable regressions; record a rejected experiment too.

**Phase gate:** The experiment is reproducible and includes a documented,
evidence-based keep/reject decision.

**Keep:** data provenance, adaptation config, base checkpoint reference,
comparison table, sample outputs, and decision notes.

**References:** [Hugging Face training](https://huggingface.co/docs/transformers/training),
[Hugging Face Datasets](https://huggingface.co/docs/datasets/).

## Phase 8: Package, serve, and operate the model

**Outcome:** A small inference application with tested inputs, documented
limits, and measured behavior.

### 8.1 Save a self-contained model bundle
- **Learn:** The relationship between weights, architecture configuration,
  tokenizer, and generation settings.
- **Do:** Bundle the selected checkpoint, configuration, tokenizer assets, and
  default generation parameters with version metadata.
- **Verify:** Load the bundle in a fresh process and generate a sample without
  relying on training-directory files.

### 8.2 Define the service contract
- **Learn:** Request/response schemas, validation, status codes, and backward
  compatibility.
- **Do:** Specify an inference request, response, supported options, and error
  behavior before adding an API framework.
- **Verify:** Write example valid and invalid requests and expected responses.

### 8.3 Implement a minimal local API
- **Learn:** Separation of transport from model logic and process lifecycle.
- **Do:** Wrap the generation API in a small local service (FastAPI is an
  optional choice). Load the model once at startup where appropriate.
- **Verify:** Start the service, make one successful request, and confirm
  errors return clear non-success responses.

### 8.4 Validate and bound resource usage
- **Learn:** Input limits, denial-of-service risk, batching, concurrency, and
  hardware memory constraints.
- **Do:** Set limits for prompt length, generated tokens, request size, and
  concurrency based on measurements.
- **Verify:** Test boundary and invalid inputs; confirm rejected requests do
  not trigger expensive generation.

### 8.5 Measure production-relevant performance
- **Learn:** Cold-start time, latency percentiles, throughput, memory, and
  realistic load tests.
- **Do:** Measure startup and inference for representative request lengths on
  the target machine.
- **Verify:** Record methodology and results. Do not report a single best-case
  latency as general performance.

### 8.6 Add operational visibility
- **Learn:** Health checks, structured logs, errors, and privacy-aware
  observability.
- **Do:** Add a health endpoint and logs for request status, duration, and
  resource-relevant metadata. Avoid logging prompt text by default.
- **Verify:** Confirm a failed model load is visible and does not appear as a
  successful response.

### 8.7 Document security and model limitations
- **Learn:** Untrusted inputs, unsafe serialized files, data handling,
  generated content risks, and intended use.
- **Do:** Document data/model provenance, limitations, prohibited or
  unsupported use, and how to report an issue. Only load artifacts from
  trusted sources.
- **Verify:** Review the service as a user: identify what data is accepted,
  stored, logged, and returned.

### 8.8 Create a reproducible release
- **Learn:** Versioning code, model artifacts, dependencies, and API behavior.
- **Do:** Tag a release or create a versioned package for the learning
  project; include setup, run, test, and shutdown instructions.
- **Verify:** Follow the README from a clean environment and check the service
  against the documented contract.

**Phase gate:** A reviewer can install, run, test, and stop the service, and
understands its resource limits and intended use.

**Keep:** model bundle manifest, API specification, tests, performance
measurements, operations guide, and model card.

**References:** [FastAPI](https://fastapi.tiangolo.com/),
[PyTorch serialization](https://pytorch.org/docs/stable/notes/serialization.html).

## Phase 9: Consolidate learning and extend toward work

**Outcome:** A portfolio-quality project that demonstrates understanding,
engineering discipline, and a deliberate next step.

### 9.1 Explain the full path
- **Do:** Trace a sample from source text through preprocessing, tokenization,
  batching, model logits, loss, checkpoint, evaluation, and generated text.
- **Verify:** Explain each stage without relying on copied definitions; draw
  the data flow and tensor shapes.

### 9.2 Reproduce the result from a clean setup
- **Do:** Follow the repository instructions in a new environment or ask
  another person to do so.
- **Verify:** Dependencies install, tests pass, a small training run finishes,
  evaluation runs, and inference loads the saved bundle.

### 9.3 Write project documentation for a teammate
- **Do:** Complete the README, experiment log, architecture notes, data
  provenance, and model card. Include commands, expected outputs, and known
  limitations.
- **Verify:** A teammate can find the answer to "How do I run it?", "What data
  was used?", and "What should I not trust it to do?" without asking you.

### 9.4 Practice a work-style change request
- **Do:** Pick one bounded request (for example, add a tokenizer test, support
  a new decoding option, or reduce inference latency). Write acceptance
  criteria, make the change in a branch, add tests, and summarize trade-offs.
- **Verify:** Review the diff as a teammate: focused scope, tests, clear
  failure handling, and documentation updated.

### 9.5 Run a controlled experiment
- **Do:** Form a hypothesis about one change (context length, model width,
  tokenizer, or learning rate). Change only that variable and preserve the
  baseline.
- **Verify:** Compare on the same held-out data and report quality, compute
  cost, and uncertainty. Explain whether the evidence supports the hypothesis.

### 9.6 Choose a specialization
- **Modeling:** Explore encoder-only models for classification or
  encoder-decoder models for sequence-to-sequence tasks.
- **Data:** Study data quality, deduplication, evaluation leakage, and
  responsible dataset curation.
- **Systems:** Explore batching, quantization, optimized attention, caching,
  and distributed training after measuring a bottleneck.
- **Applications:** Compare a small language model with retrieval-augmented
  generation or an established pretrained model for a real task.
- **Verify:** Choose the next topic based on a stated goal, not novelty alone.

**Final deliverable:** A public-safe repository (no private data or large
unreviewed artifacts) with a clear README, tested implementation, reproducible
small run, evaluation results, model card, and a concise summary of what you
learned and what remains unknown.

## Suggested milestone sequence

1. **Environment milestone:** Clean setup, tested tensor script, and working
   project structure.
2. **Data milestone:** Audited small corpus, leak-resistant splits, tokenizer,
   and verified batches.
3. **Model milestone:** Tested causal transformer with documented shapes and
   parameter count.
4. **Training milestone:** Tiny-batch overfit test, baseline comparison,
   training log, and resumable checkpoint.
5. **Evaluation/generation milestone:** Fixed evaluation suite, reported
   metrics, and tested generation API.
6. **Application milestone:** Loadable model bundle and bounded local API.
7. **Portfolio milestone:** Reproduction instructions, model card, experiment
   report, and reviewed work-style change.

## Principles to carry into real projects

- Establish a baseline and acceptance criteria before optimizing.
- Keep data provenance, model version, code revision, and configuration
  connected to every result.
- Test assumptions and edge cases, not only the happy path.
- Measure the actual bottleneck before adding performance complexity.
- Keep training, evaluation, and test data roles separate.
- Report limitations and failures as clearly as successes.
- Prefer simple, observable systems that teammates can reproduce and maintain.
