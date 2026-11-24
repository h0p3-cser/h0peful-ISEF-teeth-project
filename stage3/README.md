# Stage 3 — Quantitative light-induced fluorescence (QLF)

Analyze masked bovine-tooth QLF images at baseline (S0), 24 or 35 hours (S1), and 72 hours (S2, C04). The package produces per-image features, threshold response maps, and a same-tooth, same-view loss atlas.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ./stage3
qlf-stage3 --help
```

Python 3.10+ is required. The commands also run as `python -m qlf_stage3`. A CJK font such as Noto Sans CJK SC is needed for Chinese labels in the original plot templates.

## Input data

Set `DATA_ROOT` to the experimental workspace containing `Data/`, `Data_rgb_corrected/`, and `QLF_GPTadvice/QLF_GPTadvice_S0_S1_S2_C04.csv`. This tree supplies the capture manifest, C04 source ledger, original JPEGs, reviewed body and crown masks, registration indices, and matched RGB files. It retains historical dataset filenames. The public project includes example summary plots. Camera files, full masks and the study training CSV are local inputs or outputs.

The active data contain nine teeth, five nominal views and seven capture slots per stage/view. The reviewed T004/T005 S0 assignments are applied. The exported CSV keeps one row per active QLF image with its `sample_id`, `qlf_path`, paired `rgb_path` and `rgb_raw_path`, stage label and measurement status. Its image paths are relative to `Data/`.

## Run

```bash
qlf-stage3 --data-root "$DATA_ROOT" features --output outputs/training-features.csv
qlf-stage3 --data-root "$DATA_ROOT" area25 --output-dir outputs/area25
qlf-stage3 --data-root "$DATA_ROOT" thresholds --output-dir outputs/thresholds
qlf-stage3 --data-root "$DATA_ROOT" huber-thresholds \
  --samples outputs/thresholds/threshold-area.csv --output-dir outputs/huber-thresholds
qlf-stage3 --data-root "$DATA_ROOT" atlas --output-dir outputs/atlas
```

| Command | Outputs |
| --- | --- |
| `features` | One row per QLF image and links to the matched RGB input |
| `area25` | Overlays on target images, pixel counts and fractions, and equal-angle tooth summaries |
| `thresholds` | Direct S0→S2 curves and view panels at 5%, 10%, 20%, 30%, and 40% |
| `huber-thresholds` | S0→S1 and S0→S2 corrected curves, view panels and root-fit diagnostics |
| `atlas` | PDF, per-view PNGs, a ZIP of repeat-image panels and a pair ledger |

Ten views are used for the threshold comparison, two per angle. The thick line is the median. [Direct S0→S2](examples/direct-s2-threshold-areas.png) · [Huber S0→S1](examples/huber-s1-threshold-areas.png) · [Huber S0→S2](examples/huber-s2-threshold-areas.png).

### Mask and registration tools

- `c04-masks --qlf-body PATH --rgb-body PATH` rebuilds C04 semantic labels and indices from existing body masks. This command writes derived label files to the experimental mask directory. Run it on a separate copy of the inputs when regenerating labels.
- `direct-features --reuse-existing-crowns --output-dir PATH` regenerates paired features using reviewed crown masks. The other commands read the active pairing table from `QLF_GPTadvice/`.
- `baseline-masks --stage PATH --scope legacy --manual-root PATH` is the original baseline stain labeling workflow. It expects the archival four-folder layout and the manual seed masks. The current three-stage analysis uses the reviewed baseline index.

The earlier two-root exposure experiment is available through `root-exposure --masks PATH --semantic PATH`. It uses the archived ROI configuration and requires the corresponding legacy images, indices and hand-marked masks.

## Feature definitions

| Group | Calculation |
| --- | --- |
| Huber ΔF | Fit `target = a × S0 + b` to paired medians from disjoint 21 × 21 pixel root blocks. On the crown use `100 × (predicted − measured) / predicted`, then average the signed valid pixels. |
| 25% response area | Count valid pixels with loss strictly greater than 25%. Divide by the valid crown pixels for a percentage. Tooth summaries average repeats within a view and then weight available views equally. |
| Intensity distribution | Native green-channel mean, median, 10th percentile and population standard deviation inside the reviewed crown mask. |
| Texture | A symmetric 16-level green-channel co-occurrence matrix counts horizontal and vertical neighbor pairs wholly inside the mask. Report contrast, homogeneity, angular second moment and joint entropy. |

Longitudinal measurements require verified registration, target crown overlap of at least 85%, a common crown region of at least 500 pixels, and at least 90% usable raw signal inside it. Saturated and low-baseline pixels are excluded. The Huber prediction must be in `[16, 255)`. Baseline and failed longitudinal measurements are empty. Positive losses indicate darkening. Root holdout errors and the crown extrapolation fraction help assess the fit.

For stage classification, 917 rows have all eight single-image intensity and texture columns. Compare QLF-only and QLF+RGB using the same rows and tooth-level splits. Missing longitudinal values reveal the baseline stage. Some reviewed crown masks also depend on S0, so single-image deployment requires an independently available mask. Some source records share identical file content.

These measures describe JPEG green-channel changes, which are affected by capture conditions and stage-dependent brightness processing. The threshold area is a pixel-plane response area. Physical lesion area and mineral loss require independent validation.

## Source and tests

`src/qlf_stage3/` contains the shared package. `tests/` verifies numerical kernels using synthetic inputs.

```bash
python -m unittest discover -s stage3/tests
```

Stage 3 source code is licensed under [AGPL-3.0](LICENSE), consistent with Stage 1. Data and third-party dependencies retain their respective terms.
