# AI-Assisted SQL Query Optimization

A web-based tool that analyzes PostgreSQL queries and suggests optimizations using AI.

## Stack
- PostgreSQL (EXPLAIN / EXPLAIN ANALYZE)
- Python (backend orchestration)
- sqlparse (SQL structure analysis)
- Claude LLM (suggestions & explanations)
- Streamlit (web UI)
- A/B Testing (performance validation)

## Setup
```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Project Structure
- `app/`      → Streamlit web interface
- `core/`     → Analysis engine (validator, executor, index recommender...)
- `ai/`       → LLM client and prompt builder
- `db/`       → PostgreSQL connection management
- `security/` → Query sanitization and permissions
- `utils/`    → Logger and config
- `tests/`    → Unit and integration tests
