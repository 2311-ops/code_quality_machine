# Code Quality Machine

Phase 1 of a code-quality dataset pipeline that scrapes Python repositories from GitHub, deduplicates the results, computes Radon-based quality metrics, and saves a labeled dataset for downstream work.

## What it does

- Authenticates to GitHub with a personal access token from `.env`
- Searches GitHub repositories using the filters in `config.py`
- Collects `.py` files within the configured size bounds
- Deduplicates files by content
- Computes static metrics with Radon
- Assigns a `good`, `medium`, or `bad` quality label
- Writes the final dataset to `output/labeled_dataset.csv`

## Project Files

- `main.py` runs the full pipeline
- `scraper.py` handles GitHub API access and file collection
- `Deduplicator.py` removes duplicate code samples
- `Labeler.py` computes metrics and quality labels
- `Logger.py` sets up logging
- `config.py` stores pipeline settings and token loading

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add your GitHub token to `.env`:

```env
GITHUB_TOKEN=ghp_your_token_here
```

3. Run the pipeline:

```bash
python main.py
```

## Output

- `output/raw_code.csv` contains all scraped files before deduplication and labeling
- `output/labeled_dataset.csv` contains the final labeled dataset
- `output/scrape.log` contains the run log

## Security

- Do not commit `.env`
- Do not commit your GitHub token
- The repository already ignores `.env`, `output/`, and Python bytecode files

## Notes

- `config.py` loads `.env` automatically
- The token is read from `GITHUB_TOKEN` or `github_token`
- If no token is present, GitHub API calls fall back to the unauthenticated rate limit
