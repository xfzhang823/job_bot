# 📂 Job Bot: DuckDB Pipeline README

This document combines the **scripts guide** and the **data dictionary** into one authoritative reference for understanding and managing the DuckDB-powered job alignment pipeline.

---

## 🔍 Project Overview

This pipeline helps align resumes to job postings using structured data and LLM-driven processing. It operates through multiple stages:

- Preprocessing
- Staging
- Evaluation
- Editing
- Revaluation
- Crosstab
- LLM Trimming
- Final Export

It uses:
- ✅ File-based JSON/CSV data
- ✅ Persistent DuckDB database
- ✅ Modular scripts and Pydantic models

---

## 📦 Pipeline Scripts Overview (`job_bot/src/db_io`)

### 🔹 Create or connect to the DuckDB database
```bash
python src/db_io/create_db.py
```

### 🔹 Create all DuckDB tables
```bash
python src/db_io/setup_duckdb.py
```

---

## 📘 DuckDB Data Dictionary

All tables use a shared base schema defined in `BaseDBModel`, with the following metadata fields:

### 🔁 Shared Metadata Fields (`BaseDBModel`)
| Field         | Type               | Description |
|---------------|--------------------|-------------|
| `url`         | `HttpUrl \| str`   | Job posting URL this row belongs to. |
| `iteration`   | `int`              | Full pipeline cycle ID (starts at 0). |
| `stage`       | `PipelineStage`    | Stage that generated the row. |
| `source_file` | `Optional[str]`    | Source file path used to generate the row. |
| `timestamp`   | `datetime`         | When this row was created. |

---

## 🧱 DuckDB Table Models (by row)

### `FlattenedResponsibilitiesRow`
| Field | Type | Description |
|-------|------|-------------|
| `responsibility_key` | `str` | Key for the original resume responsibility. |
| `responsibility` | `str` | Responsibility text. |

### `FlattenedRequirementsRow`
| Field | Type | Description |
|-------|------|-------------|
| `requirement_key` | `str` | Key for the job requirement. |
| `requirement` | `str` | Requirement text. |

### `EditedResponsibilitiesRow`
| Field | Type | Description |
|-------|------|-------------|
| `responsibility_key` | `str` | Original responsibility key. |
| `requirement_key` | `str` | Target requirement key. |
| `optimized_text` | `str` | Edited responsibility text. |
| `llm_provider` | `str` | LLM provider (e.g. openai). |

### `SimilarityMetricsRow`
| Field | Type | Description |
|-------|------|-------------|
| `responsibility_key` | `str` | Key of resume responsibility. |
| `requirement_key` | `str` | Key of job requirement. |
| `responsibility` | `str` | Resume responsibility text. |
| `requirement` | `str` | Job requirement text. |
| `bert_score_precision` | `float` | Raw similarity (BERT precision). |
| `soft_similarity` | `float` | Soft token match score. |
| `word_movers_distance` | `float` | Word Mover's Distance. |
| `deberta_entailment_score` | `float` | DeBERTa entailment. |
| `roberta_entailment_score` | `float` | RoBERTa entailment. |
| *_cat fields | `Optional[str]` | Bucketed category of the above. |
| `scaled_*` fields | `Optional[float]` | Normalized versions. |
| `composite_score` | `Optional[float]` | Final weighted score. |
| `pca_score` | `Optional[float]` | PCA-reduced score. |
| `version` | `str` | Version label (e.g., original, edited). |
| `llm_provider` | `str` | Source of metrics (optional). |

### `JobPostingsRow`
| Field | Type | Description |
|-------|------|-------------|
| `status` | `str` | Scraping status. |
| `message` | `Optional[str]` | Message if any. |
| `job_title` | `str` | Title of job. |
| `company` | `str` | Employer name. |
| `location` | `Optional[str]` | Job location. |
| `salary_info` | `Optional[str]` | Salary info. |
| `posted_date` | `Optional[str]` | Posting date. |
| `content` | `Optional[str]` | Full content as JSON string. |

### `JobUrlsRow`
| Field | Type | Description |
|-------|------|-------------|
| `company` | `str` | Company name. |
| `job_title` | `str` | Job title. |

### `ExtractedRequirementsRow`
| Field | Type | Description |
|-------|------|-------------|
| `status` | `str` | Extraction status. |
| `message` | `Optional[str]` | Notes. |
| `requirement_category` | `str` | Tier (e.g., pie_in_the_sky). |
| `requirement_category_key` | `int` | Order of the category. |
| `requirement` | `str` | Text content. |
| `requirement_key` | `int` | Order in the list. |

### `PrunedResponsibilitiesRow`
| Field | Type | Description |
|-------|------|-------------|
| `responsibility_key` | `str` | Key from original set. |
| `responsibility` | `str` | Text after trimming. |
| `pruned_by` | `str` | Source of trimming (e.g., `llm`, `xfz`). |

### `PipelineState`
| Field | Type | Description |
|-------|------|-------------|
| `url` | `str` | Job posting URL. |
| `iteration` | `int` | Pipeline pass count. |
| `version` | `Literal` | One of: `original`, `edited`, `final`. |
| `status` | `Literal` | Status: `new`, `in_progress`, `complete`, `skipped`, `error`. |
| `notes` | `Optional[str]` | Optional notes (e.g., override flags). |

```
PipelineState is a control table used by the pipeline's finite state machine (FSM)
to track and advance each job (url) through its processing stages.
```

---

## 📝 Summary

All models and schemas are declared in:
- `duckdb_table_models.py`: Row-level validators
- `schema_definitions.py`: DDL and column order

The pipeline is structured, typed, and traceable. You can plug in new stages or data models with minimal changes thanks to the modular system.

---

For more detailed per-function documentation, see `duckdb_ingestion_pipeline.py` or the `run_pipelines.py` dispatch map.

