# 🚀 How to Run

Use the following command to run an experiment for Multiclass classification using only FRENCH-ITALIAN Translated DATA

```bash
python train_test.py \
    --config_source yaml \
    --yaml_path config_for_experiment.yaml \
    --data_source all \
    --strategy multiclass|pair_wise \ 
    --entity your-wandb-entity \
    --project your-project-name
```

| Argument          | Type   | Required | Choices                   | Description                                                  |
| ----------------- | ------ | -------- | ------------------------- | ------------------------------------------------------------ |
| `--config_source` | `str`  | **Yes**  | `yaml`, `sweep`           | Source of hyperparameter configuration.                      |
| `--data_source`   | `str`  | **Yes**  | `all`, `friten`           | Dataset source to use.                                       |
| `--strategy`      | `str`  | **Yes**  | `multiclass`, `pair_wise` | Training strategy.                                           |
| `--yaml_path`     | `str`  | No       | -                         | Path to YAML config file (required if `config_source=yaml`). |
| `--data_point`    | `str`  | No       | -                         | Data point identifier. Default: `'llm_both'`.                |
| `--fold`          | `str`  | No       | -                         | Cross-validation fold. Default: `'1'`.                       |
| `--up_sample`     | `flag` | No       | -                         | Enable upsampling for minority classes.                      |
| `--entity`        | `str`  | No       | -                         | Wandb entity name.                                           |
| `--project`       | `str`  | No       | -                         | Wandb project name.                                          |
| `--sweep_id`      | `str`  | No       | -                         | Wandb sweep ID (required if `config_source=sweep`).          |
| `--run_name`      | `str`  | No       | -                         | Wandb run name (required if `config_source=sweep`).          |
