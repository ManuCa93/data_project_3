# Notebook Long-Memory Notes

This README is a compact memory aid for the full notebook

## 1) Project goal

The notebook explores a large scientific-paper dataset and builds features for link prediction and citation analysis. The workflow includes:
- data loading from DuckDB / S3 parquet
- exploratory data analysis
- visualization
- missing-data analysis
- entity-resolution and normalization checks
- feature extraction and feature selection
- graph-based modeling and tabular pairwise baselines
- export of engineered datasets

## 2) Dataset characteristics

- Source: DBLP scientific papers dataset
- Scale: over 6.7 million papers
- Format: JSONL originally, then used as a parquet source in the notebook pipeline
- Main entities: papers, authors, keywords, references, venues, years, document types, languages
- Important fields used in the notebook: id, title, abstract, year, authors, keywords, references, venue, lang, doc_type, doi, page_start, page_end, n_citation

## 3) Core technical requirements

The workflow must satisfy these requirements:
- perform data exploration and visualization
- perform feature extraction and feature selection
- improve data quality by addressing missing data, including missing IDs and missing organizations
- handle entity-resolution issues, such as the same author having an organization in one year but missing it in another
- account for spelling variations in author surnames, such as Mitrovic vs Mitrović
- correct inconsistencies between venue names and document types, such as an international conference being incorrectly labeled as a journal

## 4) Notebook structure and memory map

### Data loading and stable table creation

The notebook opens a DuckDB connection and loads the parquet source from S3. It creates a stable local table called `papers` with derived fields:
- n_authors
- n_keywords
- n_references
- title_len
- abstract_len
- page_count

This table is the main working base for almost everything else.

### Exploration and visualization

The notebook includes visual checks for:
- publication counts by year
- document type distribution
- language distribution
- citation distribution
- citations by document type
- citations over time
- authors per paper over time
- prolific authors
- top keywords
- venue volume
- missingness by feature

### Missingness and data quality

The notebook explicitly measures missingness for:
- id
- title
- abstract
- keywords
- year
- authors
- references
- lang
- venue
- doc_type
- doi
- page_start
- page_end
- n_citation

It also checks missingness for nested author attributes:
- author.id
- author.org

Important reminder:
- missing author.org is much more serious than a simple null column issue because it affects entity resolution and affiliation features
- author.id should be treated as the primary identity key whenever available

### Author normalization and entity resolution

The notebook explores:
- spelling variation in author names
- normalization of names to detect likely aliases
- temporal inconsistency in author organizations

Key idea:
- normalize names by lowercasing, removing punctuation, and collapsing whitespace
- keep a small manual alias map for recurring variants
- use author.id as the stable identity when possible

### Venue and document-type consistency

The notebook also needs a cleanup mindset for metadata inconsistencies such as:
- conference incorrectly labeled as journal
- venue string variants that should map to a canonical venue name

This is important for feature engineering because venue-based signals are strong predictors.

## 5) Feature engineering memory map

The notebook builds several feature groups and then merges them into a master table called `papers_with_features`.

### Text features

From title and abstract:
- title_word_count
- abstract_word_count
- title_abstract_ratio
- title_complexity
- abstract_complexity
- has_numbers_in_title
- has_numbers_in_abstract

### Author features

- n_authors
- avg_author_citations
- max_author_citations
- multi_authored

### Network / reference features

- n_references
- references_author_ratio
- citation_richness
- has_high_ref_count

### Temporal features

- years_since_publication
- publication_era
- decade
- is_recent

### Quality / impact features

- citation_per_year
- is_highly_cited_for_year
- impact_score

### Venue / domain features

- venue
- venue_median_citations
- venue_volume
- is_top_venue
- is_english
- venue_tier

## 6) Analysis memory map

The notebook computes correlations between the target `n_citation` and engineered features, then studies interactions such as:
- venue tier vs citations
- author reputation tiers vs citations
- era vs citations
- venue tier × author tier interactions

The target variable used for modeling is:
- `n_citation`

The master feature table is:
- `papers_with_features`

## 7) Modeling memory map

### Model 1: Graph Neural Network

The notebook builds a citation graph and trains a GCN-style link predictor.
Purpose:
- predict whether one paper cites another

Inputs:
- node features from paper metadata
- citation edges from references

Evaluation:
- ROC-AUC
- Average Precision

### Model 3: Random Forest baseline for pair-level prediction

The notebook also builds a pairwise supervised dataset for citation-link prediction.

Important rule:
- avoid leakage from citation-count features like `n_citation`, `impact_score`, or other future-derived popularity signals when predicting links

Pair features include:
- year_gap
- same_year
- same_decade
- same_venue
- author_overlap
- author_jaccard
- keyword_jaccard
- title_jaccard
- abstract_jaccard
- bibliographic_coupling
- reference_jaccard
- length and count features for each paper in the pair

Evaluation and interpretation:
- ROC-AUC
- PR-AUC
- Brier score
- confusion matrix
- permutation importance
- SHAP when available

## 8) Practical reminders for future work

- Always prefer the stable `papers` table for exploratory work.
- Use `papers_with_features` for modeling and correlation analysis.
- Missing organizations and name variants are not minor issues; they affect downstream identity and collaboration features.
- Venue normalization is critical because venue prestige is one of the strongest signals.
- When predicting citation links, keep the problem pair-based and avoid leakage from global popularity metrics.
- The notebook is both an EDA report and a feature-engineering pipeline, so changes should preserve both roles.

## 9) Short memory checklist

Before continuing the notebook, remember:
1. Load data into DuckDB.
2. Build and reuse `papers`.
3. Inspect missingness and quality.
4. Normalize authors and venues.
5. Engineer text, author, network, temporal, quality, and venue features.
6. Merge everything into `papers_with_features`.
7. Correlate features with `n_citation`.
8. Train graph and pairwise models.
9. Export outputs.

## 10) Current Issues
The notebook currently suffers from a bit of an "identity crisis" and contains a massive data leakage issue in the first half, which makes it feel aimless.

Here is a breakdown of what is going well and what needs to be fixed to align with the checklist and common sense:

1. Massive Data Leakage in Models 1 & 2
The checklist explicitly states: "Target: for a pair of papers, predict if one will cite the other or not" and "keep the problem pair-based and avoid leakage from global popularity metrics."

The Problem: The first half of the notebook (Feature Engineering, Correlation Analysis, Model 1 GCN, and Model 2 XGBoost) focuses on extracting node-level features that measure a paper's overall popularity (e.g., citation_per_year, impact_score, avg_author_citations, venue_median_citations).
The Leak: When predicting if Paper A cites Paper B at the time Paper A is published, you cannot use Paper B's total citations up to 2024. Using impact_score or citation_per_year as features in the GCN and XGBoost models is a severe data leak. The model is just learning "highly cited papers are cited," which uses future information to predict a past event.
2. Model 3 (Random Forest) Does It Right
The Good: Model 3 correctly identifies the leakage problem! It pivoting to Pair-Level Features (Jaccard similarity of authors/keywords, temporal gap, etc.) without relying on any n_citation-derived metrics.
The Disconnect: Because Model 3 is so disconnected from the first half of the notebook, the first half feels completely aimless. The notebook builds up all these SQL tables for impact scores and venue tiers, only to rightfully abandon them for the Random Forest.
3. Disconnected Data Quality & Cleaning
The Problem: The SQL Data Cleaning section does some genuinely good work (e.g., author canonicalization, resolving missing IDs, tracking affiliations). However, Model 3 doesn't use any of it.
The Aimlessness: Instead of using the cleaned author_name_canonical SQL tables, the Random Forest section re-parses the raw Parquet columns using custom Python functions (extract_author_names, tokenize_text). All the hard work done in SQL to clean the data is effectively wasted because the final, valid model ignores it.
4. Misaligned EDA
The Problem: The EDA focuses on predicting single-paper popularity (e.g., "Top 15 Features Predicting Citation Count").
How to Fix: To make the notebook cohesive, the EDA should explore pairs of papers. For example:
What is the average age gap (year_gap) between a citing paper and a cited paper?
How often do citing papers share the same venue?
What is the distribution of author overlap among positive citation pairs vs. random non-citing pairs?
Summary of How to Fix the Notebook
To make this a high-quality, cohesive project that perfectly matches the checklist:

Refocus the EDA: Shift the correlation analysis away from predicting n_citation. Instead, build the pair-level dataset early and do EDA on what makes a pair a citation link vs. a non-link.
Fix or Remove Models 1 & 2: If you want to keep the GCN and XGBoost, you must remove leaking features (citation_per_year, impact_score, avg_author_citations, venue_median_citations). Replace them with leak-free node features (like title length, reference count, author count, etc.).
Connect Data Cleaning to the Pipeline: Update the pair-level feature generation (currently in Python) to utilize the cleaned SQL tables (canonical authors, normalized text) you built in the first half.

