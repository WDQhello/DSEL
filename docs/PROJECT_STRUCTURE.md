# Project Structure

## Root entry points

- `train.py`: adds `src/` to the import path, parses YAML configuration, initializes DDP, builds PACE-CLRF, and starts training.
- `test.py`: adds `src/` to the import path, initializes DDP, loads a checkpoint, and evaluates every test domain.

## `configs/`

- `default.yaml`: dataset paths, checkpoints, PACE settings, CLRF settings, optimizer settings, dataloader settings, and distributed settings.

## `scripts/`

- `train_ddp.sh`: resolves the project root, exports `PYTHONPATH`, selects GPUs, and launches distributed training.
- `test_ddp.sh`: resolves the project root, exports `PYTHONPATH`, selects GPUs, and launches distributed testing.

Both scripts can be launched from any working directory.

## `src/pace_clrf/data/`

- `transforms.py`: small-image duplication, JPEG compression, tensor conversion, and ImageNet normalization.
- `datasets.py`: dataset indexing, robust image loading, domain discovery, rank-based evaluation subsets, and collate helpers.

## `src/pace_clrf/models/`

- `lora.py`: QKV LoRA layer and automatic insertion into DINOv3 attention blocks.
- `pace.py`: Prior Information Region Enhancement using style, frequency, and noise-statistical paths.
- `clrf.py`: Dynamic Important Region Token filtering and regional prediction.
- `detector.py`: DINOv3 backbone and fusion of global and CLRF regional logits.

## `src/pace_clrf/engine/`

- `trainer.py`: distributed training, validation, gradient clipping, model selection, and checkpoint saving.
- `evaluator.py`: distributed inference, prediction gathering, per-domain metrics, and result serialization.

## `src/pace_clrf/utils/`

- `config.py`: YAML loading and command-line overrides.
- `distributed.py`: process-group initialization, rank helpers, barriers, and reductions.
- `seed.py`: deterministic random seeding and dataloader worker seeding.
- `checkpoint.py`: checkpoint extraction, DDP-prefix removal, legacy parameter-name migration, loading, and saving.
- `metrics.py`: accuracy and average precision.
- `logging.py`: rank-aware console output.

## Key design decisions

1. All hard-coded paths and hyperparameters are centralized in YAML.
2. Training and testing share one PACE-CLRF model implementation.
3. Root entry points and shell scripts work before editable package installation.
4. Validation is distributed across all ranks.
5. Evaluation avoids sampler padding so AP is not biased by duplicate samples.
6. Old checkpoints are migrated automatically to the PACE-CLRF parameter names.
7. Results are exported to CSV and JSON.
