# 📁 Dataset Overview

This directory contains training and evaluation datasets used for role classification tasks.

## 🔍 Directory Structure

```bash
ls data/all_data/
role_all/
```

### Inside `role_all`:

```bash
ls data/all_data/role_all/
llm_both/  llm_fren/  llm_iten/
```

#### Dataset Variants:

* **`llm_both/`**: Combines:

  * **D1**: Base LLM-generated data
  * **D2**: French to English translations
  * **D3**: Italian to French translations

* **`llm_fren/`**: Combines:

  * **D1**: Base LLM-generated data
  * **D2**: French to English translations

* **`llm_iten/`**: Combines:

  * **D1**: Base LLM-generated data
  * **D3**: Italian to French translations

### Example: `llm_both/` Contents

```bash
ls data/all_data/role_all/llm_both/
llm_both_test.csv
llm_both_trainEval.csv
llm_both_trainEval.json
seed42_ty_proj1_context5.csv
```

* **`llm_both_trainEval.csv`**: Combined training and evaluation set.
* **`llm_both_trainEval.json`**: Fold splits for **5-fold cross-validation**.
* **`llm_both_test.csv`**: Standard test set.
* **`seed42_ty_proj1_context5.csv`**: Gold test file with fixed seed and context configuration.
