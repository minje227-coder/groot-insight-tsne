# INSIGHT temporal t-SNE

This GitHub Pages viewer compares shared frame-level representations from four INSIGHT checkpoints and one Timewarp VAE action embedding.

## Fixed sampling contract

- Dataset: `Whalswp/INSIGHTfixposV4_filtered_multispace_v2`
- Sampling: 13 tasks x 10 randomly selected episodes per task = **130 episodes**
- RNG: one global `random.Random(42)` instance; tasks and candidate episode indices are sorted before sequential `rng.sample(..., 10)` calls
- Points: every frame from every selected episode = **6,517 t-SNE points**
- Manifest ID: `01fb32c51e`
- Videos: one full-episode clip per selected episode and camera (`guide`, `right_shoulder`, `wrist`) = **390 clips**, shared by every point and every run

Changing the seed, task order, candidate sorting, or sampling one independently seeded RNG per task creates a different manifest.

| Task | Selected episode indices | Frame points |
|---|---|---:|
| `1ext` | 3823, 228, 51, 4032, 1238, 1176, 457, 285, 4022, 209 | 323 |
| `3a` | 1912, 1945, 1845, 1610, 1868, 1782, 1582, 1581, 1613, 1677 | 387 |
| `3b` | 3051, 3190, 3240, 2945, 3219, 3033, 3211, 3146, 3044, 3161 | 379 |
| `3c` | 4329, 3769, 1040, 1118, 4244, 3801, 1116, 3737, 3799, 1089 | 378 |
| `3d` | 505, 2486, 507, 2475, 2468, 4916, 2427, 480, 4842, 4881 | 424 |
| `5a` | 2029, 2169, 2009, 2262, 2127, 2403, 2299, 2291, 2158, 2273 | 617 |
| `5b` | 3358, 3623, 3299, 3283, 3596, 3373, 3656, 3412, 3309, 3693 | 573 |
| `5c` | 4472, 4793, 4396, 4550, 4499, 4585, 4682, 4777, 4538, 4429 | 568 |
| `5d` | 763, 759, 682, 908, 712, 921, 916, 902, 611, 886 | 579 |
| `5e` | 2284, 2057, 2238, 2338, 2096, 2055, 2203, 2162, 2109, 2293 | 599 |
| `5f` | 3615, 3544, 3378, 3614, 3427, 3698, 3651, 3659, 3292, 3380 | 534 |
| `5g` | 4767, 4373, 4757, 4510, 4546, 4484, 4388, 4461, 4812, 4829 | 602 |
| `5h` | 884, 970, 753, 705, 937, 849, 799, 929, 830, 667 | 554 |
| **Total** | **130 episodes** | **6,517** |

## Families and runs

### INSIGHT Ckpt

| Run | Features | Checkpoint |
|---|---|---|
| Baseline 60K | `raw`, `processed` | `Abs_6D/Baseline/checkpoints/060000/pretrained_model` |
| LAPstyle Linear 6K 60K | `processed` | `Abs_6D/RKD_TimewarpVAE/LAPstyle_linear_6K/checkpoints/060000/pretrained_model` |
| LAPstyle Linear 10K 60K | `processed` | `Abs_6D/RKD_TimewarpVAE/LAPstyle_linear_10K/checkpoints/060000/pretrained_model` |
| RSCLstyle Cosine 60K | `processed` | `Abs_6D/RKD_TimewarpVAE/RSCLstyle_cosine_60K/checkpoints/060000/pretrained_model` |

`raw` is exported once under Baseline and reused as the shared backbone reference. Exact verification against LAP6, LAP10, and RSCL found all 493 backbone tensors and all 45 serialized pack-processor tensors bitwise equal (`value_mismatch_count=0`, `max_abs_difference=0.0`); action-space and prompt manifests also have identical SHA-256 hashes. The machine-readable report is `data/raw_equality.json`. `processed` is exported independently from every checkpoint.

### Action (Timewarp VAE)

- Run: `C8_01_C3_008 ... checkpoint-step-30000-epoch-39.11`
- Source: `action.ee_abs_rot6d`
- Per-point input: 16 future actions x 10 dimensions
- Episode tail: repeat the final valid action until all 16 steps are filled
- Feature: deterministic encoder `mu`, L2-normalized, shape `[6517, 16]`

## Feature definitions

- `raw`: attention-mask mean of `model.backbone(...)["backbone_features"]`.
- `processed`: attention-mask mean after `model.action_head.process_backbone_output(...)`.
- `action`: frozen Timewarp VAE deterministic `mu`; this is intentionally a separate family, not a GR00T action-head feature.

## Active model profiles

The sidebar shows a collapsible model profile only while that run has an active grid. Baseline `raw` and `processed` share one profile card, with both active features listed; disabling the final grid removes the card. Newly activated cards start collapsed.

The normalized profile payload is `data/training_profiles.json`. Its values are sourced from each checkpoint's `train_config.json` and `phase_schedule.json`, or from the Timewarp VAE `model_config.json` and `action_contract.json`. Policy cards keep LR warmup separate from the RKD phase range and show RKD ON/OFF, FM/RKD weights, decay schedule, distance-to-angle ratio, and trainable/frozen modules.

## t-SNE preprocessing

For each feature independently: remove constant dimensions, z-score, PCA to at most 50 dimensions, then scikit-learn t-SNE with `perplexity=30`, `max_iter=1000`, and `random_state=42`.

## Reproduction

Scripts live on DGX-H200-1 at:

```text
/home/ext_minje/clvla/benchmarks/INSIGHT/my_scripts/t-sne
```

The authoritative checkpoint root is:

```text
/home/ext_minje/groot_insight/train_ckpt/Abs_6D
```

The site output root is:

```text
/home/ext_minje/groot_insight/t-sne
```

Core commands:

```bash
python build_manifest.py
python export_videos.py --workers 12
./run_all.sh all  # fixed batch size 256 for all four checkpoint exports
./render_all.sh
```

`data/official_manifest.json` is the provenance manifest with all 6,517 rows. `data/sequences.json` is the lightweight browser payload. All run point files reference the same sequence IDs and frame indices, so a selected frame and its three videos stay aligned across comparisons.
