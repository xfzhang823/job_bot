import duckdb

# Connect to DuckDB (or create if not exists)
con = duckdb.connect("pipeline_tracking.duckdb")

# Create tables with updated schema
con.execute(
    """
CREATE TABLE IF NOT EXISTS job_postings (
    job_id TEXT PRIMARY KEY,  -- Hash of job_url for indexing
    job_url TEXT UNIQUE,      -- Original job URL for reference
    company TEXT,
    job_title TEXT,
    location TEXT,
    salary_info TEXT,
    posted_date TEXT
);
"""
)

con.execute(
    """
CREATE TABLE IF NOT EXISTS resume_tracking (
    resume_id TEXT PRIMARY KEY,
    job_id TEXT REFERENCES job_postings(job_id),
    job_url TEXT REFERENCES job_postings(job_url),
    status TEXT CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
)

con.execute(
    """
CREATE TABLE IF NOT EXISTS similarity_metrics (
    job_id TEXT REFERENCES job_postings(job_id),
    job_url TEXT REFERENCES job_postings(job_url),
    resume_id TEXT REFERENCES resume_tracking(resume_id),
    bert_score_precision FLOAT,
    soft_similarity FLOAT,
    word_movers_distance FLOAT,
    deberta_entailment_score FLOAT,
    composite_score FLOAT
);
"""
)

print("✅ Updated DuckDB schema created successfully!")
