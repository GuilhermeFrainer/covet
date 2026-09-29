# CA-BERTopic Agent Guide

The project overview, setup, pipelines, and structure live in `README.md`, imported
below so this file never drifts from it. Edit `README.md`, not this file, for
project documentation.

@./README.md

---

## Documentation Status

*   Documents directly under `docs/` describe the current repository. `docs/REPOSITORY_ISSUES.md` tracks open issues.
*   `docs/pairwise_*.md` is a partially implemented proposal; check its status line before acting on it.
*   `docs/archive/` holds completed plans and resolved incidents. Treat them as history, never as instructions.

---

## Gemini CLI Architectural Mandates & Guidelines

1.  **Test-Driven Development & Verification:**
    - Always run the full test suite (`uv run pytest`) after implementing new features, fixing bugs, or modifying configurations. Ensure all tests pass before marking tasks complete.
2.  **Linting and Formatting:**
    - Format and lint code changes using Ruff (`uvx ruff check . --fix` and `uvx ruff format .`).
3.  **Portable Documentation Links:**
    - Use relative links in markdown documentation files (e.g., `[project_structure.md](project_structure.md)` or `[src/data.py](../src/data.py)`). Never use absolute paths or `file:///` URIs in repository markdown files.
4.  **Python-Only Preprocessing Parity:**
    - All text normalization, stopword filtering, and stemming must strictly occur in Python (`src/processing.py`). R scripts must only ingest pre-processed strings from Parquet to ensure zero preprocessing discrepancies between Python models and R baselines.
5.  **Strict Row Alignment:**
    - Documents must only be retained if non-empty in **both** `clean_text` and `clean_text_stemmed`. Never filter one column independently.
6.  **Representation Stopwords Filter:**
    - Filter stop words at the c-TF-IDF topic representation layer (`CountVectorizer(stop_words="english")`) rather than stripping words from embedding inputs, preserving contextual language structure.
7.  **Result Type Isolation:**
    - Never merge or compare `standard`, `stemmed`, and `no_stopword_removal` runs into a single unsegregated analysis pool. Always respect the `--result-type` filter.
8.  **Single-Pass Archival:**
    - Ensure results merging utilities track both latest and superseded raw run files, packing them into timestamped archives and cleaning the working directory in a single pass.
9.  **Coding Style:**
    - Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
