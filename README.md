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

## 10) Split notebook map

The workflow has now been split without removing anything from the original notebook:

- [notebook.ipynb](notebook.ipynb): full original notebook, kept as the master reference
- [notebook_part1_eda_cleaning_features.ipynb](notebook_part1_eda_cleaning_features.ipynb): EDA and exploration
- [notebook_part2_feature_selection.ipynb](notebook_part2_feature_selection.ipynb): data cleaning, feature engineering, correlation analysis, interpretability, export
- [notebook_part3_modeling_link_prediction.ipynb](notebook_part3_modeling_link_prediction.ipynb): split construction, tabular baselines, pair-level modeling, graph appendix
- [notebook_helpers.py](notebook_helpers.py): shared helper functions used by the split notebooks

## 11) Current Issues — Remaining cleanup items

The major leakage paths have been addressed in the current code. What remains is mostly narrative, readability, and maintenance cleanup.

### Issue 0: Experimental design still needs a final rewrite

The notebook still does not present a fully clean final design for the tabular experiments. The flat-feature phase and the pair-level phase exist, but the model comparison story is still not fully settled.

**Impact:**
- the narrative does not yet read like a deliberate experiment plan
- it is unclear which models are the final baseline set versus temporary tests
- the notebook still mixes "strongest performance" with "model diversity" goals

**Suggested fix:**
- define a final tabular ladder explicitly before the results section
- keep one simple baseline, one strong tree baseline, and one genuinely different model family
- separate the graph-native GCN from the tabular ladder as its own phase

### Issue 1: Redundant model choices in the flat-feature phase

The current flat-feature setup still uses several very similar tree ensembles. In practice, `LGBMClassifier` and `XGBClassifier` are highly redundant, and a second tree ensemble on top of them adds limited diversity.

**Impact:**
- the model comparison is less informative than it should be
- the notebook is missing a clearly distinct baseline family
- the final results may look stronger than the real diversity of the approach

**Suggested fix:**
- keep only one of the two boosting models (`LGBMClassifier` or `XGBClassifier`)
- replace the redundant slot with a more diverse model such as:
    - `LogisticRegression` as a clean linear baseline, or
    - `MLPClassifier` if you want a genuinely different nonlinear family and are willing to standardize inputs carefully

**Practical recommendation:**
- for the notebook narrative, the most useful trio is usually:
    1. `LogisticRegression`
    2. `RandomForestClassifier` or `ExtraTreesClassifier`
    3. `LGBMClassifier`
- keep `XGBClassifier` only if you explicitly want a head-to-head booster comparison, not if diversity is the priority.

### Issue 1: Legacy experimental code is still embedded

The notebook still contains large commented-out blocks for older experiments, especially around the graph pipeline and the original XGBoost path.

**Impact:**
- harder to see which code is authoritative
- more difficult to follow the execution order
- the notebook feels like a history of attempts rather than a final report

**Suggested fix:**
- delete dead code that is no longer needed
- move historical attempts to an appendix notebook or a separate text file

### Issue 2: Section titles and model labels are inconsistent

Some headings still describe the wrong granularity or the wrong role of a model, for example using "pair" language for cells that are actually flat-feature tabular models.

**Impact:**
- the narrative is harder to follow
- readers may think a model is pair-level when it is actually node/flat-level

**Suggested fix:**
- rename sections to something explicit like:
    - Flat tabular baselines
    - Pair-level baseline
    - Graph-native model
- keep numbering consistent across the three model blocks

### Issue 3: The notebook is doing too many jobs in one flow

EDA, missing-data analysis, entity resolution, feature engineering, graph construction, flat-model training, pair-model training, and interpretation are all interleaved in one long notebook.

**Impact:**
- readability drops
- execution order becomes fragile
- it is harder to explain the notebook as a coherent story

**Suggested fix:**
- split into two notebooks if possible:
    1. EDA + cleaning + feature engineering
    2. modeling + interpretation
- if keeping one notebook, add a short table of contents and stronger section separators

### Issue 4: Outputs need a final refresh

Some cells still show outputs from earlier runs and earlier versions of the code.

**Impact:**
- old outputs can contradict the current code
- readers may trust stale permutation-importance or error outputs

**Suggested fix:**
- rerun from the flat feature-construction cell onward
- clear obsolete outputs before the final handoff

### Issue 5: Leakage policy should be centralized

The flat-feature pipeline now excludes the leaky global-statistics columns correctly, but the exclusion policy is still spread across multiple cells.

**Impact:**
- future edits can accidentally reintroduce leakage
- the notebook is harder to maintain

**Suggested fix:**
- define one clear constant for node/flat exclusions near the data-loading section
- reuse it everywhere the flat matrix is built

### Issue 6: Target-derived feature leakage in flat-feature engineering (CRITICAL)

The current flat-feature matrix does not mainly leak through scaling; it leaks through the feature definitions themselves. Several node-level inputs are computed from citation counts, venue prestige, or author reputation statistics that are not available at citation-prediction time.

**Code locations:**
- author reputation features: [notebook.ipynb](notebook.ipynb#L886-L904)
- quality features: [notebook.ipynb](notebook.ipynb#L1590-L1708)
- flat feature export / selection: [notebook.ipynb](notebook.ipynb#L1746-L1773)

**Examples of risky features:**
- `avg_author_citations`
- `max_author_citations`
- `citation_per_year`
- `impact_score`
- `venue_median_citations`
- `is_top_venue`
- `venue_tier`

**Impact:**
- the model can learn from post-publication popularity signals instead of publication-time metadata
- performance becomes inflated and less meaningful for true link prediction
- even with temporal filters like `year <= 2020`, the features may still encode future citation knowledge

**Suggested fix:**
- keep these features only in a clearly separated "analysis-only" branch, if at all
- for the link-prediction baseline, rebuild the flat matrix from publication-time-safe metadata only
- make the allowed feature list explicit and document why each feature is safe

### Issue 7: Non-stratified pair sampling across train/val/test (MODERATE)

Pair-level features subsample pairs randomly inside each split. The split boundaries exist, but the same source paper can still appear across train/val/test because the sampling is not group-aware.

**Code location:** Cell 119, lines 2857–2897:
```python
pos_idx = np.random.choice(len(pos_src), n_pos, replace=False)
neg_idx = np.random.choice(len(neg_src), n_neg, replace=False)
```

**Impact:**
- train and test can share source papers or nearby citations
- evaluation may be slightly optimistic if the same paper-level context repeats across splits
- the current protocol is better described as split-consistent sampling, not strict group separation

**Suggested fix:**
- if strict independence is required, enforce group-aware splitting by source paper or publication year
- otherwise, document that the split is edge-based and not group-exclusive
- add an overlap check before training

### Issue 8: Inconsistent dataset sizes across experimental phases (MODERATE)

The three modeling phases use radically different dataset sizes, making fair comparison impossible.

**Impact:**
- **Phase 1 (Flat features):** ~1.5M train pairs (FRACTION_TRAIN=0.05)
- **Phase 2 (Pair-level RF):** only 5k train pairs randomly sampled from Phase 1
- **Phase 3 (GCN):** uses full original graph
- model performance differences may be due to dataset size, not model quality
- impossible to isolate model family contribution from data volume effect

**Suggested fix:**
- either use the same pair counts across all tabular phases (flat + pair-level)
- or explicitly document why phases use different scales
- rerun all three phases on the same ~100k pair sample for a fair head-to-head comparison

### Issue 9: Poor documentation of feature exclusion policy (MODERATE)

The `leaky_features` list is defined at cell 94 (lines 1900–1912), but then referenced via `.getattr(globals(), 'leaky_features', [])` at cell 95 (line 2677). This is fragile and error-prone.

**Code location:**
- Cell 94: list definition
- Cell 95, line 2677: unsafe retrieval via globals()
- Cell 98, line 2967: uses the list

**Impact:**
- future maintainers may not realize that `leaky_features` must be defined before cell 95
- if cells are reordered or re-executed out of order, features silently reintroduce leakage
- no single source of truth for the exclusion policy
- pair-level feature engineering (cells 119–121) never explicitly documents which node signals are forbidden

**Suggested fix:**
- define `LEAKY_FEATURES` as a module-level constant in an early setup cell (after data loading, before feature engineering)
- document the rationale for each excluded feature in a clear comment
- verify at each use site that the exclusion list was applied
- explicitly document which pair-level features are derived only from publication-time metadata

### Issue 10: Permutation importance is completely disabled (MINOR)

Cell 110 (lines 2691–2720) has the entire permutation importance block commented out, with no explanation in surrounding markdown.

**Impact:**
- no feature importance analysis for the flat-feature Random Forest
- readers cannot see which pair-level signals drive citations
- interpretation is incomplete

**Suggested fix:**
- either delete the commented code if it's truly obsolete
- or uncomment, document why it was disabled, and schedule it to run at the end (it takes ~80 min)
- provide a markdown cell explaining the choice

### Issue 11: Large commented-out historical code blocks (MINOR)

Cell 105 (lines 2428–2520) contains 90+ lines of commented "ORIGINALE" XGBoost code. Similar commented blocks exist elsewhere.

**Impact:**
- clutters the notebook
- confuses readers about which code is authoritative
- makes diffs harder to read during version control

**Suggested fix:**
- delete all commented historical code blocks
- if historical versions are important, save them to a separate file or commit to git history
- keep only active, running code in the main notebook
