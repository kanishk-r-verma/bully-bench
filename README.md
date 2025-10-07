# emnlp-industry-track
This is the anonymous github repo for EMNLP industry track submission

## Data Availability

This repository includes experiments based on data from  **Italian-to-English** and  **French-to-English** translation and **𝓓<sub>Gold</sub>** also  **LLM-generated dataset**. 

## Access Notice
Due to ongoing review, both the **translations** and **𝓓<sub>Gold</sub>** with **LABELS** will be made available **after the review process is complete**.
Although for reproducing **Extrinsic** and **Intrinsic** Evaluation  **translation datasets**,  **LLM - generated** unlabelled data are made available only with **role** labels.
We appreciate your understanding and patience.

---
## 📂 Directory Structure and Purpose

- **`data/`**
  - **`GOLD_TEST.csv`**
     BullyBench --> GOLD_TEST  
  - **`all_data/role_all/`**  
    Used as input by `train_test.py`. See the `src/` directory for training and testing scripts.
  - **`intrinsic/`**  
    Contains the annotation data used by `human_eval_final.ipynb` for intrinsic evaluation.

- **`src/`**
  - **`requirements.txt`**
    make sure to install requirements.txt as per your setup (conda, pip etc)
  - **`dependencies.py`**
  - **`*.yaml`**
    see hyperparameter yaml config table
  - **`utils.py`**
  - **`train_test.py`**

- **`supplementary_files/`**
  - Contains supporting files such as **annotation guides (PDFs)** and other documentation to assist in understanding and reproducing the evaluation process.

---
## Extrinsic Evaluation

### RoBERTa-base & GPT2-medium Top-2 on 𝓓<sub>Gold</sub>

| Model       | Train-Eval Data                       | Strategy   | Loss Function  | Upsampling? | Fold | Macro-F1 | Enabler F1 | Defender F1 | Bully F1 | Victim F1 |
|-------------|----------------------------------------|------------|----------------|-------------|------|----------|-------------|--------------|-----------|------------|
| RoBERTa     | **𝓓₁ ∪ 𝓓₂ ∪ 𝓓₃**                        | Pair-wise  | Focal Loss     | True        | 2    | 0.4034   | 0.4422      | 0.6052       | 0.3420    | 0.2243     |
| RoBERTa     | **𝓓₁ ∪ 𝓓₂ ∪ 𝓓₃**                        | Multiclass | Cross Entropy  | True        | 4    | 0.3911   | 0.5156      | 0.5700       | 0.3085    | 0.1702     |
| GPT2-medium | **𝓓₁ ∪ 𝓓₂ ∪ 𝓓₃**                        | Multiclass | Focal Loss     | True        | 4    | 0.3807   | 0.5234      | 0.5775       | 0.2317    | 0.1905     |
| GPT2-medium | **𝓓₁ ∪ 𝓓₂ ∪ 𝓓₃**                        | Pairwise   | Cross Entropy  | True        | 1    | 0.3756   | 0.5000      | 0.5519       | 0.2919    | 0.1587     |

To Reproduce Extrinsic Evaluation Results mentioned in Sections 2.6, 3.2, 4 and Appendix D (D.1, D.2, and D.3) follow YAML Config to To Model + Strategy Mapping and Directory Structure + Execution steps


### Hyper-parameter configurations for **BASELINE** on  𝓓<sub>Gold</sub>

| YAML Filename                                 | Model        | Strategy    | Loss Function   | Fold |
|----------------------------------------------|--------------|-------------|------------------|------|
| `roberta_pair_upsample_focal_fold2_all.yaml` | RoBERTa      | Pair-wise   | Focal Loss       | 2    |
| `roberta_mclass_upsample_cross_fold4_all.yaml`| RoBERTa      | Multiclass  | Cross Entropy    | 4    |
| `gpt2_mclass_upsample_focal_fold4_all.yaml`  | GPT2-medium  | Multiclass  | Focal Loss       | 4    |
| `gpt2_pair_upsample_cross_fold1_all.yaml`    | GPT2-medium  | Pairwise    | Cross Entropy    | 1    |


### How to Run

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


---
## Intrinsic Evaluation

### 📘 Notebooks

- **`human_eval_final.ipynb`**  
  This notebook is designed to **calculate and analyze responses** from the **Intrinsic Evaluation** process for **five distinct annotators** across **six dimensions of cyberbullying**.

---

### 🧑‍💻 Annotators and Roles

| User              | Annotator Type       |
|-------------------|----------------------|
| OnlineObserver    | Social Scientist     |
| SafetyEvaluator   | Social Scientist     |
| StreamerDreamer   | Content Moderator    |
| HideNShare        | Content Moderator    |
| CloudSafeZone     | Adult Teen           |

---

Feel free to reach out for any clarifications regarding the setup or usage.
