# 🛠️ Scripts Guide – `job_bot/src/db_io`

Welcome to your command center. This file documents the most useful scripts, commands, and common workflows for working with the `job_bot` project.

---

## 🧠 Project Overview

This project uses:
- ✅ File-based pipelines (`output/json/`, `metrics/`, etc.)
- ✅ Persistent DuckDB backend (`pipeline_data/db/pipeline_data.duckdb`)
- ✅ Script-based helpers for DB creation, JSON ingestion, and analysis

---

## 📦 DuckDB Utilities

### 🔹 Create or connect to the database
```bash
python src/db_io/create_db.py
