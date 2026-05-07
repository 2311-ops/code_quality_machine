# Code Quality Machine

An end-to-end code-quality pipeline that scrapes Python repositories from GitHub, deduplicates the results, computes static-analysis features, trains ML models, and exposes a Phase 4 inference helper for raw Python code.

## What it does

- Authenticates to GitHub with a personal access token from `.env`
- Searches GitHub repositories using the filters in `config.py`
- Collects `.py` files within the configured size bounds
- Deduplicates files by content
- Computes static metrics with Radon
- Assigns a `good`, `medium`, or `bad` quality label
- Builds a labeled dataset and feature matrix in `output/`
- Trains and saves baseline ML models and a 1D CNN classifier
- Provides a reusable Phase 4 inference function for raw Python code

## Project Files

- `main.py` runs the full pipeline
- `scraper.py` handles GitHub API access and file collection
- `Deduplicator.py` removes duplicate code samples
- `Labeler.py` computes metrics and quality labels
- `feature_engineering.ipynb` builds the feature matrix and train/test splits
- `ML_modeling.ipynb` trains and evaluates the ML models
- `phase4_inference.py` loads the saved model and returns a quality report for raw code
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
- `output/features_dataset.csv` contains the full engineered feature matrix
- `output/model_cnn.keras` contains the saved 1D CNN model
- `output/model_metadata.json` contains the saved model metadata used by Phase 4
- `output/scaler.json` stores the feature scaling parameters
- `output/tfidf_vocab.json` stores the TF-IDF vocabulary
- `output/scrape.log` contains the run log

## Phase 4 Inference

Use `phase4_inference.py` when you need to score a raw Python snippet and return a JSON-ready quality report:

```python
from phase4_inference import analyze_python_code_quality

report = analyze_python_code_quality("def add(a, b):\n    return a + b\n")
print(report["prediction"]["label"])
print(report["quality_report"]["summary"])
```

The report includes:

- predicted label and confidence
- class probabilities
- extracted metrics
- suggestions and a score
- model metadata from `output/model_metadata.json`

## Security

- Do not commit `.env`
- Do not commit your GitHub token
- The repository already ignores `.env`, `output/`, and Python bytecode files

## Notes

- `config.py` loads `.env` automatically
- The token is read from `GITHUB_TOKEN` or `github_token`
- If no token is present, GitHub API calls fall back to the unauthenticated rate limit

## Documentation

- [`docs/project_rubric.md`](docs/project_rubric.md) contains the project evaluation rubric captured from the reference image
