# Multimodal Toxicity Detection

This repository contains the code used for a bachelor's thesis on multimodal
toxicity detection in online spaces. The project is built around image-and-text
classification tasks such as hateful meme detection and social-media
hate-speech evaluation.

The repository is used to prepare datasets, build an aggregated Hateful Memes +
PrideMM dataset, optionally generate image captions, train multimodal
classifiers, evaluate checkpoints, and generate supporting reports. The active
model families documented here are CLIP, CLIP align-fusion (similar to [Hate-CLIper](https://github.com/gokulkarthik/hateclipper)), ViLT, BLIP-2, and BLIP-2 align-fusion.

## Getting Started

Datasets are not stored in the repository. Download and unpack them under `data/`
before running training or evaluation.

```bash
mkdir -p data
```

Use these dataset sources:

| Dataset       | Download                                                                                           | Expected location     |
| ------------- | -------------------------------------------------------------------------------------------------- | --------------------- |
| Hateful Memes | [Kaggle](https://www.kaggle.com/datasets/williamberrios/hateful-memes?select=hateful_memes)        | `data/hateful_memes/` |
| PrideMM       | [Google Drive](https://drive.google.com/file/d/17WozXiXfq44Z6kkWsPPDHRzqIH2daUaQ/view?usp=sharing) | `data/PrideMM/`       |
| MMHS150K      | [Google Drive](https://drive.google.com/file/d/1S9mMhZFkntNnYdO-1dZXwF_8XIiFcmlF/view?usp=sharing) | `data/MMHS150K/`      |

The default training and evaluation workflow can run directly on Hateful Memes,
but the thesis workflow used the aggregated dataset after it was generated. Even
when training on a specific source dataset after aggregation, runs used
`data/aggregated` with `--source hateful_memes` or `--source pridemm` instead of
switching back to the original source roots.

After placing the datasets, sync the Python environment:

```bash
uv sync
```

Install optional extras only when needed:

```bash
uv sync --extra captions
uv sync --extra analysis
uv sync --extra captions --extra analysis
```

Show the CLI help with:

```bash
uv run python main.py --help
```

## Dataset Layouts

### Hateful Memes

Unpack Hateful Memes into `data/hateful_memes/`. Training and evaluation expect a
Hateful-Memes-style root with JSONL metadata and an image directory.

```text
data/hateful_memes/
  img/
  train.jsonl
  dev_seen.jsonl
  test_seen.jsonl
  test_unseen.jsonl
```

### PrideMM

Unpack PrideMM into `data/PrideMM/`. The aggregation script expects:

```text
data/PrideMM/
  Images/
  PrideMM.csv
```

PrideMM is used by this repository through the aggregated dataset builder.

### MMHS150K

Unpack MMHS150K into `data/MMHS150K/`. Evaluation expects resized images and the
metadata JSON file:

```text
data/MMHS150K/
  img_resized/
  MMHS150K_GT.json
```

MMHS150K is mainly used for evaluation-style experiments. When a 2-class
checkpoint is evaluated on MMHS150K, labels are binarized as `0 = not hate` and
`1-5 = hate`.

## Aggregated Dataset

The aggregated dataset combines Hateful Memes and PrideMM into a binary
dataset under `data/aggregated/`. It is the main dataset root
used in the thesis experiments after the aggregation step.

Run a dry run first to validate inputs and inspect the planned summary:

```bash
uv run python scripts/build_aggregated_dataset.py --dry-run
```

Build or rebuild the dataset with:

```bash
uv run python scripts/build_aggregated_dataset.py --overwrite
```

The script writes:

```text
data/aggregated/
  img/
  train.jsonl
  val.jsonl
  test.jsonl
  index.jsonl
  manifest.json
```

`train.jsonl`, `val.jsonl`, and `test.jsonl` contain the normalized records used
for training and evaluation. `index.jsonl` preserves source provenance, original
image paths, and source-specific labels. `manifest.json` records source roots,
split policy, source names, resize policy, and dataset statistics.

All model-facing records use binary labels with `0 = not hate` and `1 = hate`.
Copied images are assigned synthetic sequential filenames under
`data/aggregated/img/`.

### Split Policy

The aggregation split policy is fixed:

| Aggregated split | Source splits                                                                      |
| ---------------- | ---------------------------------------------------------------------------------- |
| `train`          | Hateful Memes `train`, decontaminated Hateful Memes `test_unseen`, PrideMM `train` |
| `val`            | Hateful Memes `dev_seen`, PrideMM `val`                                            |
| `test`           | Hateful Memes `test_seen`, PrideMM `test`                                          |

### PrideMM Image Resizing

By default, the aggregated builder resizes copied PrideMM images above 2,000,000
total pixels while preserving aspect ratio. This only affects PrideMM copies
under `data/aggregated/img/`; original PrideMM files are not modified.

Disable this behavior with:

```bash
uv run python scripts/build_aggregated_dataset.py --overwrite --pridemm-max-pixels 0
```

### Source Filtering

Aggregated training and evaluation can be filtered with `--source`. Valid source
names are read from `data/aggregated/manifest.json`.

For thesis-style source-specific training, keep `data/aggregated` as the data
root and apply the source filter:

```bash
uv run python main.py train clip-align data/aggregated --source hateful_memes
uv run python main.py train clip-align data/aggregated --source pridemm
```

The filter applies to train, validation, and post-training test splits during
training. During single-checkpoint evaluation, it applies to the selected
aggregate split.

## Captions

Captions are optional. Training and evaluation work without them, and captions
are disabled by default. Pass `--captions` only when running an explicit
caption-fusion experiment.

The captions used in the thesis are already available in `captions/`:

```text
captions/
  hateful_memes_captions.json
  pridemm_captions.json
  aggregated_captions.json
```

Caption files are stored at repository level, not inside dataset directories.
For a dataset root named `data/hateful_memes/`, the matching caption file is
`captions/hateful_memes_captions.json`. For `data/aggregated/`, the matching file
is `captions/aggregated_captions.json`.

When `data/aggregated` is built, the aggregation script remaps final source
caption JSON files to the synthetic aggregated image paths and writes
`captions/aggregated_captions.json` when source captions are available. It reads
only final JSON files, not temporary caption progress JSONL files.

### Generate New Captions

Caption generation uses a local `llama.cpp` server exposing an OpenAI-compatible
`v1/chat/completions` endpoint. The thesis captions were generated with
`Qwen3.5-9B-UD-Q8_K_XL.gguf`. If you want to reproduce or experiment with this
setup, the model is available from [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF).

Start the server with:

```bash
./scripts/start_llama_server.sh --model ~/models/Qwen3.5-9B-UD-Q8_K_XL.gguf
```

The server script disables reasoning by default with `--reasoning-budget 0`. To
allow reasoning, pass a nonzero budget or `-1` for no limit:

```bash
./scripts/start_llama_server.sh --model ~/models/Qwen3.5-9B-UD-Q8_K_XL.gguf --reasoning-budget -1
```

By default, the script expects the multimodal projector at
`~/models/qwen35-mmproj-F16.gguf`. Override it when your local filename differs:

```bash
./scripts/start_llama_server.sh \
  --model ~/models/Qwen3.5-9B-UD-Q8_K_XL.gguf \
  --mmproj ~/models/qwen35-mmproj-F16.gguf
```

Extra raw `llama-server` flags can be forwarded after `--`:

```bash
./scripts/start_llama_server.sh --model ~/models/Qwen3.5-9B-UD-Q8_K_XL.gguf -- --verbose
```

Generate captions with:

```bash
uv run python main.py caption ./data/hateful_memes --image-dir img
uv run python main.py caption ./data/PrideMM --image-dir Images
uv run python main.py caption ./data/aggregated --image-dir img
uv run python main.py caption ./data/MMHS150K --image-dir img_resized
```

The command scans the image directory recursively. Progress is written to a
temporary JSONL file next to the final output while captioning is running. After
the final JSON file is written successfully, the temporary file is deleted.

Caption generation error logs are timestamped under `reports/logs/`. Images are
decoded locally before requests; formats not directly sent to llama.cpp, such as
WebP files, are converted to PNG bytes in memory without modifying source files.

Use `--debug-responses` to write raw llama.cpp JSON responses to
`<output>.responses.jsonl` for successful requests and caption-response errors.

### Default Prompt

```text
Task: Produce a concise, factual caption of the image for research use.

Describe all salient visual elements, including objects, people, actions, visible text, and important spatial relationships. Use neutral, objective language and avoid interpretation, speculation, or moral judgment.

If people are present, describe each individual separately using observable attributes only. Include apparent gender presentation, apparent race/ethnicity, and approximate age category (for example: child, teenager, adult, elderly). If any attribute is not visually clear, state that it is unclear rather than guessing. Do not infer identity, intent, beliefs, or internal states.

Describe any actions taking place explicitly.

If the image contains sexual, violent, or otherwise explicit content, describe it factually and precisely without euphemism or censorship.

Transcribe all visible text exactly as it appears. If text is partially unreadable, include the readable portion and mark the unclear part as unclear.

Respond with a single caption only. Do not include explanations, disclaimers, bullet points, or content warnings.
```

### Default Decoding Parameters

- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- `max_tokens=4096`
- `seed=42`

These values are sent per request by the captioning client. They were chosen for
Qwen3.5, and reasoning is disabled unless `--reasoning` is passed to the caption
command and the server was started with a nonzero reasoning budget.

## Training

The default data root is `data/hateful_memes/` if no root is passed. For the
thesis workflow, use `data/aggregated` after building the aggregated dataset.

Train on the full aggregated dataset:

```bash
uv run python main.py train clip data/aggregated
uv run python main.py train clip-align data/aggregated
uv run python main.py train vilt data/aggregated
uv run python main.py train blip2 data/aggregated
uv run python main.py train blip2-align data/aggregated
```

Train on a source-specific slice of the aggregated dataset:

```bash
uv run python main.py train clip data/aggregated --source hateful_memes
uv run python main.py train clip data/aggregated --source pridemm
```

Captions are disabled by default. Add `--captions` when a matching caption file
exists and you want caption fusion:

```bash
uv run python main.py train clip-align data/aggregated --captions
```

Training uses early stopping on validation loss. The main controls are:

- `--max-epochs`: hard upper bound on training length
- `--patience`: stop if validation loss does not improve for this many epochs
- `--min-delta`: minimum validation-loss improvement required to reset patience
- `--checkpoint-limit`: checkpoint retention policy
- `--checkpoint-strategy`: `best-per-metric` by default, or `best-loss`

With `--checkpoint-strategy best-per-metric`, training keeps the current best
loss, accuracy, and AUROC checkpoints. `--checkpoint-limit` is ignored in this
mode. With `--checkpoint-strategy best-loss`, `--checkpoint-limit -1` saves every
epoch and `--checkpoint-limit N` keeps the best `N` validation-loss checkpoints.

After training finishes, the best validation-loss checkpoint is reloaded
automatically. The saved best loss, accuracy, and AUROC checkpoints are then
evaluated on the test split when available.

Training, validation, and post-training test logs are timestamped under
`reports/logs/`. Checkpoints are written under
`ckpt/<model_name>/<YYYYMMDD_HHMM>/`, for example
`ckpt/CLIPClassifier/20260517_1438/best-loss_epoch3.pt`.

Each checkpoint run directory also contains `metadata.json`, recording dataset
root, source filter, data source, caption settings, and model architecture
caption-fusion details.

The main dataloader defaults are `--batch-size 64`, `--prefetch-factor 2`, and
`--no-pin-memory`. `--pin-memory` is available as an opt-in flag, but it can
substantially increase normal RAM usage because images are decoded as
full-resolution float tensors before model-specific preprocessing.

### Model Notes

`clip` uses OpenCLIP by default with `--clip-model-name ViT-L-14` and
`--clip-pretrained datacomp_xl_s13b_b90k`. Its classifier input concatenates
L2-normalized image embeddings and text embeddings.

`clip-align` is a Hate-CLIPper-style OpenCLIP align-fusion experiment. It maps
projected OpenCLIP image/text embeddings to `--map-dim 1024`, L2-normalizes them,
multiplies them elementwise, and applies `--num-pre-output-layers 3` hidden
pre-output layers with `--pre-output-dim 1024`. Its training defaults are
`--lr 1e-4`, `--weight-decay 1e-4`, AdamW, and `--gradient-clip-val 0.1`.

`vilt` uses frozen Hugging Face ViLT features followed by a classifier head. It
defaults to `--vilt-model-name dandelin/vilt-b32-mlm-itm`; the processor files
for this default are loaded from `dandelin/vilt-b32-mlm`. ViLT text is truncated
to `--max-text-length` BERT tokens, with a default of `40`.

`blip2` uses frozen BLIP-2 image/text retrieval features followed by a classifier
head.

`blip2-align` uses the shared align-fusion head with frozen BLIP-2 pooled vision
features and pooled Q-Former text features. BLIP-2 processing is memory-heavy, so
use conservative batch size, worker count, and prefetching on memory-limited
machines.

Normal CLIP, `clip-align`, ViLT, BLIP-2, and `blip2-align` checkpoints are
separate families and are not interchangeable. Older align-fusion checkpoints
trained with one pre-output layer should be evaluated with
`--num-pre-output-layers 1`.

When `--captions` is used, captions are encoded as a separate third modality with
frozen ModernBERT. Caption text is not appended to the original dataset text.

## Evaluation

Single-checkpoint evaluation supports the active model families:

```bash
uv run python main.py eval clip ckpt/CLIPClassifier/<run>/best-loss_epoch3.pt
uv run python main.py eval clip-align ckpt/CLIPAlignFusionClassifier/<run>/best-auroc_epoch10.pt
uv run python main.py eval vilt ckpt/ViLTClassifier/<run>/best-loss_epoch5.pt
uv run python main.py eval blip2 ckpt/BLIP2Classifier/<run>/best-loss_epoch3.pt
uv run python main.py eval blip2-align ckpt/BLIP2AlignFusionClassifier/<run>/best-loss_epoch3.pt
```

Evaluation defaults to the validation split. Pass `--split test` to evaluate the
test split. For Hateful Memes, validation means `dev_seen` and test means
`test_seen`. For the aggregated dataset, these are `val` and `test`.

Evaluate a checkpoint on `data/aggregated` with:

```bash
uv run python main.py eval clip ckpt/CLIPClassifier/<run>/best-loss_epoch3.pt data/aggregated --split test
```

Filter aggregated evaluation by source with:

```bash
uv run python main.py eval clip ckpt/CLIPClassifier/<run>/best-loss_epoch3.pt data/aggregated --split val --source hateful_memes
uv run python main.py eval clip ckpt/CLIPClassifier/<run>/best-loss_epoch3.pt data/aggregated --split test --source pridemm
```

For binary tasks, evaluation reports loss, accuracy, and AUROC. This also applies
to MMHS150K when evaluated in the binarized `not hate` vs `hate` setting.

Single-checkpoint evaluation writes timestamped logs under `reports/logs/`.

### Modality Ablation

To ablate one modality during evaluation, pass `--drop-modality image` or
`--drop-modality text`. Image ablation replaces image tensors with zeros. Text
ablation replaces text inputs with empty strings. When image ablation is used,
captions are disabled automatically to avoid leaking image-derived caption text.

```bash
uv run python main.py eval clip ckpt/CLIPClassifier/<run>/best-loss_epoch3.pt --drop-modality image
uv run python main.py eval clip ckpt/CLIPClassifier/<run>/best-loss_epoch3.pt --drop-modality text
```

### Cleaned MMHS150K Subset

Build the cleaned MMHS150K subset with:

```bash
uv run python scripts/build_mmhs150k_clean.py data/MMHS150K
```

The script writes `data/MMHS150K/MMHS150K_clean.json`. Evaluate it by passing the
generated metadata filename:

```bash
uv run python main.py eval clip ckpt/CLIPClassifier/<run>/best-loss_epoch3.pt data/MMHS150K --metadata-file MMHS150K_clean.json
```

When a non-default MMHS metadata file is used, all records in that file are
evaluated as one split.

## Reports

Generate a Hateful Memes split-overlap diagram with:

```bash
uv run --extra analysis python scripts/plot_hateful_memes_overlaps.py
```

The script writes `reports/hateful_memes_overlaps.png`. It draws a single
Euler-style diagram over `train`, `dev_seen`, `dev_unseen`, `test_seen`, and
`test_unseen` using sample ID overlap, while checking that image filename overlap
has the same structure.

Generated run logs live under `reports/logs/` and are ignored by git. Other
report artifacts, such as `reports/hateful_memes_overlaps.png`, can be
regenerated from scripts.
