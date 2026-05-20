# Code Quality Machine

An end-to-end code-quality pipeline that scrapes Python repositories from GitHub, deduplicates the results, computes static-analysis features, and trains ML models.

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
- `Logger.py` sets up logging
- `config.py` stores pipeline settings and token loading

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

TensorFlow is only needed for the training notebooks and saved model work, not for the Streamlit UI.
If you want to rerun the deep-learning notebook locally, install it separately in that environment.

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
- `output/model_metadata.json` contains the saved model metadata
- `output/scaler.json` stores the feature scaling parameters
- `output/tfidf_vocab.json` stores the TF-IDF vocabulary
- `output/scrape.log` contains the run log

## UI

Run the Streamlit UI:

```bash
streamlit run app.py
```

The app provides:

- a dashboard with project statistics and screenshots
- a code analysis workflow for pasted or uploaded Python files
- a model comparison and artifact download page
- a short project walkthrough covering all three phases

## Deploying to Streamlit Cloud

This repository is ready to publish on Streamlit Community Cloud.

1. Push the project to GitHub.
2. In Streamlit Cloud, create a new app and point it at this repository.
3. Set the main file path to `app.py`.
4. Add `GITHUB_TOKEN` as an app secret only if you want the scraping pipeline to use the GitHub API from the deployed environment.

The deployed app will open directly with:

```bash
streamlit run app.py
```

## Security

- Do not commit `.env`
- Do not commit your GitHub token
- The repository already ignores `.env`, `output/`, and Python bytecode files

## Notes

- `config.py` loads `.env` automatically
- The token is read from `GITHUB_TOKEN` or `github_token`
- If no token is present, GitHub API calls fall back to the unauthenticated rate limit

## Documentation

- [`CodeQualityAssessor_Documentation.docx`](CodeQualityAssessor_Documentation.docx) is the main project documentation
