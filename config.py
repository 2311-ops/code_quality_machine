"""Phase 1 configuration."""

from pathlib import Path
import os


def _load_dotenv(path: Path) -> None:
    """
    Lightweight .env loader for simple KEY=VALUE lines.

    It keeps the project dependency-free while still allowing local
    secrets to live in .env during development.
    """
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv(Path(__file__).with_name(".env"))

GITHUB_TOKEN =  os.getenv("github_token", "")

# -- Search Filters --
SEARCH_QUERIES = [
    "language:python stars:>200 size:<50000",
    "language:python stars:>500 topic:data-science",
    "language:python stars:>300 topic:web",
    "language:python stars:>300 topic:cli",
]

MAX_REPOS_PER_QUERY = 10        # repos to scrape per search query
MAX_FILES_PER_REPO  = 30        # max .py files to pull from each repo
MIN_FILE_BYTES      = 500       # skip trivially small files
MAX_FILE_BYTES      = 50_000    # skip massive generated files

# -- Quality Labeling Thresholds --
# Maintainability Index (0-100): higher = more maintainable
MI_GOOD_THRESHOLD   = 65
MI_BAD_THRESHOLD    = 40
# Average Cyclomatic Complexity: lower = simpler
CC_GOOD_THRESHOLD   = 5
CC_BAD_THRESHOLD    = 10

# -- Output --
OUTPUT_DIR          = "output"
RAW_CSV             = "output/raw_code.csv"
LABELED_CSV         = "output/labeled_dataset.csv"
LOG_FILE            = "output/scrape.log"
