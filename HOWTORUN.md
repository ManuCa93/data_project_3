# Citation Link Prediction: From Tabular Baselines to Graph Neural Networks

**Authors:** Alessio Carnevale, Manuel Cattoni, Carlo Schillaci

This notebook tackles the task of **scientific citation link prediction**: given a pair of academic papers, predict whether one will cite the other. The pipeline spans Exploratory Data Analysis, Feature Engineering, and three modeling phases — tabular baselines (Random Forest, LightGBM, MLP), pair-link feature models, and a Graph Convolutional Network (GCN) with hard negative sampling.

---

## Prerequisites

Install all required dependencies before running either option:

```bash
pip install duckdb polars pandas numpy scikit-learn lightgbm torch torch-geometric matplotlib seaborn python-dotenv
```

---

## Option 1 — Run the Full Notebook from Scratch

This option executes the entire pipeline: data ingestion from S3, EDA, feature engineering, temporal splitting, and model training.

### 1. Switch to a local Parquet file

If you have a local copy of the dataset, open the notebook and update cell 2:

```python
# PARQUET = "s3://data-project-supsi-bucket/data.parquet"
PARQUET = "data.parquet"  # ← uncomment this line
```

### 2. Run all cells

Open the notebook in Jupyter and run all cells sequentially from top to bottom:

```bash
jupyter notebook notebook_def_fixed.ipynb
```

Or using JupyterLab:

```bash
jupyter lab notebook_def_fixed.ipynb
```

The pipeline will:
1. Ingest and query the Parquet dataset via DuckDB
2. Perform EDA and data quality checks
3. Engineer node-level and pair-level features
4. Perform a temporal split (train ≤ 2020, val 2021–2022, test ≥ 2023)
5. Save preprocessed splits to `model_splits/`
6. Train and evaluate all models

> ⚠️ **Note:** Querying 6.7M records from S3 is time- and bandwidth-intensive. Expect the data ingestion and feature engineering cells to take significant time depending on your connection and hardware.

---

## Option 2 — Download Pre-computed Splits and Run from Cell 86

This option skips the data ingestion, EDA, and feature engineering entirely. You download the ready-to-use train/val/test splits and jump straight to model training.

### 1. Download the splits

Download the pre-computed split files from Google Drive:

📁 **[Download model_splits from Google Drive](https://drive.google.com/drive/folders/1ESxAuHjjl1MVhbRbvjv5HWepOFBORCRW?usp=sharing)**

The archive contains:

```
model_splits/
├── train.parquet       # Papers published up to 2020
├── val.parquet         # Papers published 2021–2022
├── test.parquet        # Papers published 2023 and later
```

### 2. Place the files in the correct location

Extract or move the downloaded files so they sit in a `model_splits/` folder **in the same directory as the notebook**:

```
your-project-folder/
├── notebook.ipynb
└── model_splits/
    ├── train.parquet
    ├── val.parquet
    ├── test.parquet
```

### 3. Open the notebook and run from Cell 86

Open the notebook in Jupyter:

```bash
jupyter notebook notebook_def_fixed.ipynb
```

Navigate to **Cell 86** — titled *"Load Splits and Train Models"* — and run all cells from there to the end. You can do this via the menu:

> **Cell → Run All Below**

Cell 86 onwards covers:
- Importing all modelling libraries (LightGBM, PyTorch, scikit-learn)
- Loading the split Parquet files with Polars
- **Phase 1:** Flat tabular baselines (Random Forest, LightGBM, MLP)
- **Phase 2:** Pair-link feature models
- **Phase 3:** Graph Convolutional Network (GCN) with hard negative sampling

> ℹ️ Cells 84–85 (which generate the splits) will be automatically skipped — they detect that the `model_splits/` files already exist and print a confirmation message without recomputing anything.

---

## Project Structure

```
your-project-folder/
├── notebook_def_fixed.ipynb   # Main notebook
├── .env                       # AWS credentials (Option 1 only)
├── papers_with_features.parquet  # Intermediate artifact (generated in Option 1)
└── model_splits/              # Train/val/test splits (generated or downloaded)
```

## Pipeline Overview

| Stage | Cells | Description |
|---|---|---|
| Setup & Ingestion | 1–5 | DuckDB connection, S3 config, table creation |
| EDA | 6–50 | Distributions, missingness, citation analysis |
| Canonicalization | 51–70 | Venue normalization, author deduplication |
| Feature Engineering | 71–83 | Text, author, network, and temporal features |
| Splitting & Scaling | 84–85 | Temporal split + StandardScaler |
| **Model Training** | **86–124** | **← Start here for Option 2** |
