Project Overview
A machine learning tool that scrapes live Python repositories from GitHub, extracts static code metrics, trains a classifier to assess code quality, and generates human-readable improvement suggestions. Built from scratch — no pre-made datasets (no Kaggle, no dataset downloads).
Stack: Python 3.12 · scikit-learn · TensorFlow/Keras · PyGithub · Radon · Flask/FastAPI · Streamlit

Project Roadmap
PhaseTitleStatusDuration1Data Collection✅ Complete~1–2 weeks2Feature Engineering⬜ Not started~1 week3ML Modeling⬜ Not started~2 weeks4Tool & Interface⬜ Not started~1 week5Evaluation & Report⬜ Not started~1 week

Phase 1 — Data Collection
What it does
Scrapes live .py files from GitHub using the REST API, deduplicates them by content hash, computes static quality metrics via Radon, and saves a labeled CSV ready for Phase 2 feature engineering.
Directory structure
phase1/
├── main.py            ← pipeline entry point — run this
├── config.py          ← all tunable settings live here
├── scraper.py         ← GitHub API scraping logic
├── labeler.py         ← Radon metrics + quality label assignment
├── deduplicator.py    ← MD5 content-hash deduplication
├── logger.py          ← shared console + file logger
└── output/
    ├── raw_code.csv       ← all scraped files before labeling
    ├── labeled_dataset.csv← final labeled training data
    └── scrape.log         ← full timestamped run log
How to run
bash# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your GitHub token in your shell
$env:GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 3. Run the pipeline
cd phase1/
python main.py
Getting a GitHub token

Go to github.com/settings/tokens
Click Generate new token (classic)
Set expiration to 90 days
Check only: ✅ public_repo
Copy the token — it is shown only once
Set the `GITHUB_TOKEN` environment variable before running.


Security: Never commit the token. Use os.getenv("GITHUB_TOKEN") and a .env file in production.


Module reference
config.py — Settings
All tunable parameters. Edit this before running.
import os
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()

SEARCH_QUERIES = [             # GitHub search queries to run
    "language:python stars:>200 size:<50000",
    "language:python stars:>500 topic:data-science",
    "language:python stars:>300 topic:web",
    "language:python stars:>300 topic:cli",
]

MAX_REPOS_PER_QUERY = 10       # repos scraped per query
MAX_FILES_PER_REPO  = 30       # max .py files per repo
MIN_FILE_BYTES      = 500      # skip trivially small files
MAX_FILE_BYTES      = 50_000   # skip huge generated files

MI_GOOD_THRESHOLD   = 65       # Maintainability Index: good cutoff
MI_BAD_THRESHOLD    = 40       # Maintainability Index: bad cutoff
CC_GOOD_THRESHOLD   = 5        # avg Cyclomatic Complexity: good cutoff
CC_BAD_THRESHOLD    = 10       # avg Cyclomatic Complexity: bad cutoff

OUTPUT_DIR   = "output"
RAW_CSV      = "output/raw_code.csv"
LABELED_CSV  = "output/labeled_dataset.csv"
LOG_FILE     = "output/scrape.log"

scraper.py — GitHub API
FunctionDescriptionconnect()Authenticates with GitHub, logs remaining rate limit_wait_for_rate_limit(g)Blocks until the API rate limit resets (auto-called)_get_python_files(repo, g)Recursively walks a repo's file tree, returns list of file dictsscrape_all(g)Runs all search queries, collects all files, returns flat list
File dict schema (one entry per .py file):
python{
  "repo":      "owner/repo-name",
  "repo_url":  "https://github.com/owner/repo-name",
  "stars":     1234,
  "file_path": "src/utils.py",
  "file_size": 2048,          # bytes
  "code":      "def foo(): ..."
}
Rate limit behaviour: Authenticated = 5,000 req/hr. Unauthenticated = 60 req/hr. The scraper auto-detects a RateLimitExceededException and sleeps until reset — no manual intervention needed.

labeler.py — Quality Metrics + Labeling
Uses Radon to compute three categories of metrics per file:
Metrics computed:
ColumnSourceDescriptionmi_scoreradon.metrics.mi_visitMaintainability Index (0–100). Higher = more maintainableavg_ccradon.complexity.cc_visitAverage Cyclomatic Complexity across all functionsmax_ccradon.complexity.cc_visitComplexity of the most complex functionlocradon.raw.analyzeTotal lines of codellocradon.raw.analyzeLogical lines of codeslocradon.raw.analyzeSource lines (no blanks/comments)commentsradon.raw.analyzeNumber of comment linesblankradon.raw.analyzeNumber of blank linescomment_ratiocomputedcomments / lloc
Label assignment logic:
MI >= 65  AND  avg_cc <= 5   →  "good"
MI <  40  OR   avg_cc > 10   →  "bad"
everything else               →  "medium"
Files that fail Radon parsing are silently dropped and logged.
FunctionDescription_compute_metrics(code)Returns metric dict for one code string, or None on parse failure_assign_label(mi, avg_cc)Returns "good" / "medium" / "bad"label_dataframe(df)Applies metrics + labels to a full DataFrame, drops unparseable rows

deduplicator.py — Content Deduplication
Removes identical files (common due to GitHub forks) using MD5 hashing of raw code content.
pythondeduplicate(df: pd.DataFrame) -> pd.DataFrame
Adds a temporary content_hash column, drops duplicates, removes the column, and logs how many rows were removed.

logger.py — Shared Logger
pythonget_logger(name: str) -> logging.Logger
Returns a logger that writes simultaneously to stdout and output/scrape.log. All modules use this — do not use print() in pipeline code.

main.py — Pipeline Orchestrator
Runs the four steps in order:
Step 1 → scraper.scrape_all()        → raw_code.csv
Step 2 → deduplicator.deduplicate()  → removes duplicate files
Step 3 → labeler.label_dataframe()   → adds metrics + quality_label
Step 4 → save labeled_dataset.csv    → ready for Phase 2
Prints a summary at the end:
Total files scraped    : 312
After deduplication    : 287
After labeling         : 281
  good   :  94 files  (33.5%)
  medium : 121 files  (43.1%)
  bad    :  66 files  (23.5%)

Output schema — labeled_dataset.csv
ColumnTypeDescriptionrepostrowner/reporepo_urlstrFull GitHub URLstarsintStar count at scrape timefile_pathstrPath within the repofile_sizeintFile size in bytesmi_scorefloatMaintainability Index (0–100)avg_ccfloatAverage cyclomatic complexitymax_ccintMax cyclomatic complexitylocintTotal linesllocintLogical linesslocintSource linescommentsintComment line countblankintBlank line countcomment_ratiofloatcomments / llocquality_labelstrgood / medium / badcodestrRaw source code (for Phase 2 tokenization)

Tuning tips
If you get too many "medium" labels: tighten the thresholds:
pythonMI_GOOD_THRESHOLD = 70   # raise good bar
MI_BAD_THRESHOLD  = 45   # raise bad bar
CC_GOOD_THRESHOLD = 4
CC_BAD_THRESHOLD  = 8
If you want more data: increase per-query limits or add more queries:
pythonMAX_REPOS_PER_QUERY = 20
SEARCH_QUERIES += ["language:python stars:>100 topic:machine-learning"]
If you hit rate limits frequently: reduce the scrape speed:
pythontime.sleep(3)  # in scraper.py → scrape_all()

Dependencies
PyGithub>=2.1.1     # GitHub REST API client
radon>=6.0.1        # static code metrics
pandas>=2.0.0       # DataFrame manipulation
requests>=2.31.0    # HTTP (used internally by PyGithub)
Install:
bashpip install PyGithub radon pandas requests

Phase 2 — Feature Engineering (upcoming)
The labeled_dataset.csv produced by Phase 1 feeds directly into Phase 2. Planned feature extraction:

Already available (from Phase 1): mi_score, avg_cc, max_cc, loc, comment_ratio
To be added in Phase 2:

AST-based features via Python's ast module (nesting depth, function count, import count)
Token-level TF-IDF vectors via scikit-learn
Optional: CodeBERT embeddings via HuggingFace transformers



Tools: ast (stdlib), pyflakes, scikit-learn, transformers

Notes

No pre-made datasets are used. All data is collected live from public GitHub repositories.
The scraper biases toward quality code by filtering for repos with >200 stars.
Files under 500 bytes (trivial) and over 50KB (likely generated/minified) are excluded.
The code column is preserved in the labeled CSV for downstream tokenization in Phase 2.
All pipeline steps are idempotent — re-running main.py overwrites output files cleanly.
