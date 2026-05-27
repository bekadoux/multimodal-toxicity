# Project Implementation Overview

This report describes the implementation in the multimodal toxicity detection
repository. VisualBERT-related code is intentionally excluded because that model
family is slated for removal.

## Purpose

The project implements thesis experiments for multimodal toxicity detection in
online meme/image-text datasets. The core workflow trains and evaluates frozen
vision-language backbones with lightweight classifier heads on binary hate vs.
non-hate data, with support for image-caption augmentation, source-specific
aggregated-dataset analysis, and modality ablations.

The repository is built around Python 3.12, `uv`, PyTorch, torchvision,
Transformers, OpenCLIP, scikit-learn, and tqdm. Optional analysis dependencies
include matplotlib and numpy. The full dependency resolution is pinned in
`uv.lock`, and PyTorch/torchvision use the CUDA 13.0 index on Linux and Windows.

## Entry Point

`main.py` defines a single argparse CLI with three top-level command groups:

- `train`: trains supported model families.
- `eval`: evaluates a single checkpoint on validation or test data.
- `caption`: generates repository-level image caption JSON files through a local
  llama.cpp server.

The training commands covered here are `clip`, `clip-align`, `vilt`, `blip2`, and
`blip2-align`. The matching evaluation commands use the same model-family names
and load one checkpoint path. The CLI defaults to `data/hateful_memes/` for
training and evaluation unless a different data root is passed.

Shared dataloader defaults are conservative: batch size `64`, prefetch factor
`2`, pin memory disabled, and a CPU-derived worker count capped at `4`. Captions
are enabled by default when the matching caption file exists. Aggregated datasets
can be filtered with `--source`, and single-checkpoint evaluation can select
`--split val` or `--split test`.

## Repository Layout

- `core/`: shared training, evaluation, checkpoint I/O, caption client, and log
  helpers.
- `dataset/`: dataset implementations, collators, datamodules, class-weight
  computation, source validation, and train/eval datamodule factories.
- `models/`: CLIP, CLIP align-fusion, ViLT, BLIP-2, and BLIP-2 align-fusion
  model implementations.
- `commands/`: CLI command implementations that assemble datamodules, models,
  losses, optimizers, logging, checkpoint metadata, and evaluation routines.
- `scripts/`: dataset-building and reporting utilities, plus the llama.cpp
  server launcher.
- `captions/`: generated caption JSON artifacts stored at repository level.
- `ckpt/`: checkpoint output directory, kept empty in git except `.gitkeep`.
- `data/`: expected local dataset roots, kept empty in git except `.gitkeep`.
- `reports/`: generated figures, ignored run logs, and this overview.

## Dataset Support

`dataset/dataset.py` implements all dataset readers used by the project.

`ImageTextJsonlDataset` is the base reader for Hateful-Memes-style binary JSONL
datasets. It expects split files such as `train.jsonl`, resolves image paths
under an `img/` directory, reads images with torchvision, converts them to float
tensors in `[0, 1]`, normalizes grayscale and alpha-channel images to RGB, and
returns `(text, image, label)`. If a caption JSON is loaded, it appends
`IMG_CAPTION: ...` to the text using the image path relative to the dataset root.
Missing labels are returned as `-1` for ignored evaluation examples.

`HatefulMemesDataset` is a thin specialization of `ImageTextJsonlDataset` using
the standard Hateful Memes layout. The default split mapping is `train`,
`dev_seen` for validation, and `test_seen` for test.

`AggregatedDataset` extends the JSONL reader with an optional `source` filter.
It supports aggregated split files named `train.jsonl`, `val.jsonl`, and
`test.jsonl`. If a source filter is requested and no records from that source
exist in a split, it raises a clear error.

`MMHS150KDataset` supports MMHS150K evaluation-style experiments. It loads a
metadata JSON file, reads image tensors from `img_resized/<tweet_id>.jpg`, adds
OCR text from `img_txt/<tweet_id>.json` when present, appends captions when
available, and returns annotator vote tensors. With the default metadata file it
uses split ID files under `splits/`; with a non-default metadata file it can use
all records as one evaluation set.

## Datamodules

`dataset/datamodule.py` provides lightweight datamodule classes and factory
functions.

`BinaryImageTextDataModule` handles Hateful-Memes-style and aggregated binary
data. It caches train, validation, and test dataloaders; uses a collate function
that keeps variable-sized images as a list and stacks scalar labels; moves images
and labels to the target device in `process_batch`; and computes inverse-frequency
class weights from the training split.

`HatefulMemesDataModule` configures the binary datamodule for standard Hateful
Memes split names.

`AggregatedDataModule` configures the binary datamodule for aggregated split
names and passes source filtering into `AggregatedDataset`.

`MMHSDataModule` supports MMHS150K train/validation/test loader construction for
evaluation paths. Its `process_batch` converts annotator votes to a majority
label. In binary mode, MMHS labels are binarized as `0 = not hate` and all
nonzero labels as `1 = hate`; six-class mode preserves labels `0` through `5`.

`build_train_data_module` auto-detects aggregated roots by `manifest.json` plus
`train/val/test.jsonl`, and Hateful Memes roots by `train/dev_seen/test_seen`.
Training on MMHS150K is not exposed through this factory.

`build_eval_data_module` supports MMHS150K roots with `img_resized/`, aggregated
roots, and Hateful Memes roots. It validates that `--source` is only used with
aggregated data and reads allowed source names from the aggregated manifest.

## Shared Training

`core/train.py` implements the training loop.

`train_epoch` requires a model-specific `process_batch` callback, runs a standard
forward/backward/optimizer step, tracks loss and accuracy, supports optional
gradient-norm clipping, and appends periodic loss entries to a train log.

`train_model` validates training controls, creates timestamped log paths when
needed, creates a checkpoint run directory under `ckpt/<model_name>/<YYYYMMDD_HHMM>/`,
writes optional `metadata.json`, and runs epoch-level training and validation. It
uses validation loss for early stopping with `--patience` and `--min-delta`, while
also tracking best validation loss, accuracy, and AUROC.

With the default `best-per-metric` checkpoint strategy, training retains one
current-best checkpoint for each metric: `best-loss`, `best-accuracy`, and
`best-auroc`. Older checkpoints for the same metric are deleted when a better one
appears. With `best-loss`, checkpoint retention is controlled by
`--checkpoint-limit`; `-1` keeps every epoch checkpoint.

At the end of training, the best-loss checkpoint is reloaded into the model,
best-metric summaries are printed and logged, and the paths for retained best
checkpoints are returned to the command layer.

## Shared Evaluation

`core/eval.py` implements evaluation and metric reporting.

`evaluate` runs the model in eval mode with a required `process_batch` callback,
ignores labels equal to the criterion `ignore_index` value, computes average
loss, accuracy, a scikit-learn classification report, a confusion matrix, and
binary AUROC when the model has two logits and both classes are present. Metrics
are printed and optionally appended to a log file.

`evaluate_best_checkpoints` reloads each retained best checkpoint into a freshly
constructed model and evaluates it on a supplied dataloader. Training commands
use this for post-training test evaluation when a test loader is available.

`commands/eval_utils.py` selects validation or test dataloaders and implements
modality ablation. Text ablation replaces each text input with an empty string.
Image ablation replaces images with zero tensors and disables captions to avoid
leaking image-derived information through appended caption text. Processor-backed
models use a collator wrapper so ablation happens before ViLT or BLIP-2
preprocessing.

## Checkpoints And Logs

`core/io.py` creates checkpoint run directories, saves checkpoint metadata, saves
model and optimizer state dictionaries, and loads checkpoint files with optional
`map_location`. Checkpoint files include the epoch plus any available validation
loss, accuracy, and AUROC values.

`core/logs.py` creates timestamped log filenames under `reports/logs/`, sanitizes
filename components, and ensures the log directory exists. Training commands
write train, validation, and post-training test logs. Evaluation commands write
single-checkpoint eval logs.

`commands/train_metadata.py` records checkpoint metadata describing the dataset,
data root, source filter, caption request status, whether captions were actually
used, and the caption file path when applicable.

## CLIP Classifier

`models/clip_classifier.py` implements the default CLIP classifier.

The feature extractor loads a frozen OpenCLIP model and preprocessing transform,
defaults to `ViT-L-14` with `datacomp_xl_s13b_b90k`, tokenizes text with the
matching OpenCLIP tokenizer, preprocesses each image tensor, extracts image and
text embeddings with `encode_image` and `encode_text`, L2-normalizes both
embedding sets, and concatenates them.

The classifier head applies layer normalization, linear layers with GELU and
dropout through hidden widths `512`, `256`, and `128`, then a final classification
layer. The training command optimizes only the classifier head parameters with
AdamW, class-weighted cross entropy, and the default learning rate `1e-5`.

## Shared Align-Fusion Head

`models/align_fusion.py` implements reusable align-fusion logic shared by CLIP
align-fusion and BLIP-2 align-fusion.

`AlignFusionFeatureExtractor` maps frozen image and text features separately to a
common `map_dim`, applies dropout, L2-normalizes both mapped vectors, and returns
their elementwise product. Backbone-specific subclasses implement
`extract_features`.

`AlignFusionClassificationHead` applies an initial fusion dropout, then a
configurable number of linear/ReLU/dropout pre-output layers, and finally a
classification layer. `num_pre_output_layers` may be zero, but not negative.

`AlignFusionClassifier` composes a feature extractor with the classification
head. This keeps the fusion implementation independent of the underlying
backbone.

## CLIP Align-Fusion Classifier

`models/clip_align_fusion_classifier.py` implements the CLIP-backed align-fusion
experiment.

The model loads a frozen OpenCLIP backbone, extracts projected image and text
features with `encode_image` and `encode_text`, maps both to `map_dim`, multiplies
the normalized mapped features elementwise, and classifies the fused feature.
The default CLI settings are `map_dim=1024`, `pre_output_dim=1024`, three
pre-output layers, map dropout `0.1`, fusion dropout `0.4`, pre-output dropout
`0.2`, AdamW, learning rate `1e-4`, weight decay `1e-4`, and gradient clipping
at `0.1`.

Normal CLIP and CLIP align-fusion checkpoints are separate model families and are
not interchangeable.

## ViLT Classifier

`models/vilt_classifier.py` implements the ViLT classifier.

The input processor wraps `ViltProcessor` and runs in the DataLoader collator. It
defaults to model `dandelin/vilt-b32-mlm-itm`, but uses processor files from
`dandelin/vilt-b32-mlm` for that default because the mlm-itm repository ships
only weights and config. It pads and truncates text to `max_text_length`, defaults
to `40`, and uses `do_rescale=False` because dataset images are already float
tensors.

The frozen `ViltModel` backbone exposes either the raw final CLS hidden state or
the model pooler output. The default is `cls`. The classifier head mirrors the
project's classifier-head pattern with layer normalization, GELU, dropout, a
configurable first hidden width, then `256`, `128`, and output logits.

Training and evaluation use `process_vilt_batch` helpers to move processor output
tensors to the device and to convert MMHS vote lists into majority labels when
needed. When captions are requested, commands print and log a warning that long
caption-augmented text may be truncated by ViLT tokenization.

## BLIP-2 Classifier

`models/blip2_classifier.py` implements the BLIP-2 classifier.

The BLIP-2 input processor wraps `Blip2Processor` and runs in a custom collator,
moving CPU-side image preprocessing and tokenization into DataLoader workers. It
uses `do_rescale=False` and channels-first image tensors from the dataset.

The frozen backbone loads `Blip2ForImageTextRetrieval`, defaults to
`Salesforce/blip2-itm-vit-g`, and uses bfloat16 on CUDA and float32 on CPU. The
vision path returns the BLIP-2 vision model `pooler_output`. The text path runs
the Q-Former over token embeddings, strips image-query tokens when required by
the model config, and returns token features plus attention masks.

The classifier pools image features directly, mean-pools text features with the
attention mask, concatenates the image and text vectors, and applies a
layer-normalized MLP classifier. The default training command uses AdamW, learning
rate `1e-5`, weight decay `1e-3`, and class-weighted cross entropy.

## BLIP-2 Align-Fusion Classifier

`models/blip2_align_fusion_classifier.py` implements BLIP-2-backed align-fusion.

It reuses the frozen BLIP-2 backbone and feature extractors from the normal BLIP-2
classifier, but sends pooled image and pooled Q-Former text features into the
shared align-fusion path. The default training command uses the same align-fusion
defaults as CLIP align-fusion, including learning rate `1e-4`, weight decay
`1e-4`, and gradient clipping at `0.1`.

Normal BLIP-2 and BLIP-2 align-fusion checkpoints are separate model families and
are not interchangeable with each other or with CLIP-family checkpoints.

## Training Commands

Each covered training command follows the same structure:

- Select `cuda` when available, otherwise CPU.
- Disable tokenizer parallelism with `TOKENIZERS_PARALLELISM=false`.
- Build and set up the train datamodule.
- Compute binary training class weights from the train split.
- Build checkpoint metadata including dataset and caption details.
- Instantiate the model with CLI-supplied backbone and head settings.
- Use class-weighted `CrossEntropyLoss(ignore_index=-1)`.
- Use AdamW over trainable parameters.
- Create train, validation, and test log paths with a shared timestamp.
- Call `train_model` with early stopping and checkpoint strategy controls.
- Evaluate retained best checkpoints on the test loader when one exists.

The CLIP command optimizes only the classifier head because the OpenCLIP backbone
is frozen. The align-fusion, ViLT, and BLIP-2 commands optimize all trainable
parameters, which are the heads and projection layers because their backbones are
frozen.

## Evaluation Commands

Each covered evaluation command follows the same structure:

- Select `cuda` when available, otherwise CPU.
- Create an eval log under `reports/logs/`.
- Prepare caption loading and modality ablation settings.
- Build and set up the eval datamodule.
- Select the requested validation or test dataloader.
- Instantiate the matching model architecture and head configuration.
- Load the checkpoint into that model.
- Evaluate with `CrossEntropyLoss(ignore_index=-1)` and the model-specific
  `process_batch` callback.
- Print and log loss, accuracy, and AUROC when available.

Evaluation defaults to the validation split. For Hateful Memes that means
`dev_seen`; for the aggregated dataset it means `val`. Passing `--split test`
uses Hateful Memes `test_seen` or aggregated `test`.

## Caption Generation

`core/captions.py` and `commands/generate_captions.py` implement image captioning
through a local llama.cpp server exposing an OpenAI-compatible
`/v1/chat/completions` endpoint.

Caption files are named from the dataset root as
`captions/<dataset_name>_captions.json`. Image keys are relative paths from the
dataset root, such as `img/00000001.png` or `Images/img_1.png`.

The default prompt asks for concise, factual image descriptions, visible text
transcription, and neutral observable attributes. Default decoding parameters are
temperature `0.6`, top-p `0.95`, top-k `20`, max tokens `4096`, seed `42`, and
reasoning disabled unless requested.

The caption client embeds image bytes as base64 data URLs. JPEG and PNG are sent
directly. Other decodable image formats, such as WebP, are converted to PNG in
memory; source files are not modified. Transparent images are composited onto a
white background before PNG encoding.

The generation command discovers images recursively or under a specified
`--image-dir`, loads existing final captions and temporary progress JSONL,
skips already captioned images unless `--overwrite` is used, appends every new
caption to progress, optionally writes full debug responses, retries missing
captions once, fails if any captions remain missing, writes the final JSON
atomically, and removes the progress JSONL after success.

`scripts/start_llama_server.sh` launches `llama-server` with default Qwen3.5 GGUF
and multimodal projector paths under `~/models`, validates those paths, exposes
host/port/context/GPU-layer/thread/parallel/reasoning controls, and forwards extra
raw llama-server arguments after `--`.

## Aggregated Dataset Builder

`scripts/build_aggregated_dataset.py` builds a binary Hateful-Memes-style
aggregated dataset from Hateful Memes and PrideMM. HarMeme may exist locally, but
this script does not include it.

Inputs default to `data/hateful_memes`, `data/PrideMM`, and output
`data/aggregated`. The output layout is `img/`, `train.jsonl`, `val.jsonl`,
`test.jsonl`, `index.jsonl`, and `manifest.json`.

The fixed split policy is Hateful Memes `train`, decontaminated Hateful Memes
`test_unseen`, and PrideMM `train` for aggregate train; Hateful Memes `dev_seen`
and PrideMM `val` for aggregate validation; and Hateful Memes `test_seen` and
PrideMM `test` for aggregate test.

Labels are normalized to binary `0` and `1`. The training JSONL records contain
synthetic sequential IDs, synthetic image paths under aggregate `img/`, text,
binary label, source, source ID, and source split. `index.jsonl` preserves source
image paths and source-specific labels for provenance.

The decontamination step is intentionally narrow. Before Hateful Memes
`test_unseen` records are added to aggregate train, records are removed only when
their normalized text overlaps aggregate validation or test text. Normalization
uses `casefold`, stripping, and whitespace collapse. Hateful Memes `train`,
PrideMM `train`, validation, and test records are not otherwise decontaminated.

The builder remaps available source caption JSON files into
`captions/aggregated_captions.json` by matching each original source image path
to its new synthetic aggregate image path. It reads final caption JSON files only,
not temporary progress JSONL files.

PrideMM images above `--pridemm-max-pixels` are resized by default to at most
2,000,000 pixels while preserving aspect ratio with Pillow LANCZOS. Original
PrideMM files are unchanged; only copied aggregate images may be resized. Passing
`--pridemm-max-pixels 0` disables resizing.

The script supports `--dry-run` to validate inputs and print planned summaries
without writing files. Non-dry runs require `--overwrite` when output paths
already exist.

## MMHS150K Clean Subset Builder

`scripts/build_mmhs150k_clean.py` builds a deterministic cleaned MMHS150K metadata
file for evaluation.

The script loads `MMHS150K_GT.json`, excludes records without exactly three
annotator labels, excludes any record with a religion vote, and keeps only
records with unanimous `3/3` or majority `2/3` agreement. It samples fixed quotas:
4,000 non-hate examples and 1,000 examples each for racist, sexist, homophobe,
and other-hate classes. Sampling is deterministic through SHA-256-derived stable
seeds from a base seed, default `1337`.

The output is `MMHS150K_clean.json`, where each selected record is enriched with
`majority_label`, `majority_label_str`, and `agreement`.

## Hateful Memes Overlap Report

`scripts/plot_hateful_memes_overlaps.py` generates `reports/hateful_memes_overlaps.png`.

The script loads Hateful Memes `train`, `dev_seen`, `dev_unseen`, `test_seen`, and
`test_unseen` split files, builds sample-ID and image-filename sets, validates
that the only ID overlap is between `dev_seen` and `dev_unseen`, warns if image
filename overlap differs from ID overlap, and draws a fixed Euler-style diagram
with matplotlib using the non-interactive `Agg` backend.

## Generated Artifacts

The tracked caption JSON files under `captions/` are generated image-key to
caption maps for Hateful Memes, PrideMM, and the aggregated dataset. They are data
artifacts consumed by dataset readers and the aggregated builder, not executable
implementation code.

`reports/hateful_memes_overlaps.png` is a generated figure from the overlap
plotting script. Timestamped training, evaluation, and caption logs live under
`reports/logs/`, which is ignored by git.

`data/` and `ckpt/` are intentionally kept out of git except for `.gitkeep`
files. Local datasets, checkpoints, and large derived artifacts are expected to
remain local.

## Current Boundaries

There is no committed test suite. Verification guidance in the repository favors
small smoke tests, CLI help checks, targeted dataloader checks, and one-batch
model checks rather than full training runs.

Training through the main factory supports Hateful Memes-style and aggregated
binary datasets. MMHS150K is supported mainly for evaluation and cleaned-subset
experiments.

All main backbones are frozen. The project trains classifier heads and, for
align-fusion models, projection and fusion layers on top of frozen backbone
features.

Caption generation requires a running local llama.cpp server with multimodal
support. The repository provides the client and launcher, but not the model files.

Checkpoints are architecture-specific. CLIP, CLIP align-fusion, ViLT, BLIP-2, and
BLIP-2 align-fusion checkpoints cannot be mixed across model families.
