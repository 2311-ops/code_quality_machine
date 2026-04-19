"""
scraper.py
Scrapes Python files from GitHub repos using the PyGithub API.
"""

from inspect import stack
import time

from github import Github, GithubException, RateLimitExceededException

from config import (
    GITHUB_TOKEN,
    MAX_FILE_BYTES,
    MAX_FILES_PER_REPO,
    MAX_REPOS_PER_QUERY,
    MIN_FILE_BYTES,
    SEARCH_QUERIES,
)
from Logger import get_logger

log = get_logger("scraper")


def _get_core_rate(g: Github):
    """
    Return the core API rate bucket from PyGithub's rate-limit overview.
    """
    #ask github what is my current rate limit status
    overview = g.get_rate_limit()
    #
    resources = getattr(overview, "resources", None)
    if resources is not None and getattr(resources, "core", None) is not None:
        return resources.core
    return overview.rate


def connect() -> Github:
    #if token is provided, use it to authenticate and get a higher rate limit
    if GITHUB_TOKEN:
        g = Github(GITHUB_TOKEN)
        user = g.get_user().login
        #check how many API requests we have remaining in the current rate limit window
        remaining = _get_core_rate(g).remaining
        
        log.info(f"Authenticated as '{user}' - {remaining} API requests remaining")
    else:
        g = Github()
        log.warning("No token provided - limited to 60 requests/hour")
    return g

#function to wait until the rate limit resets, with a buffer of 5 seconds to be safe
def _wait_for_rate_limit(g: Github) -> None:
    """Block until the rate limit resets."""
    #get the current rate limit status
    rl = _get_core_rate(g)
    #github provides a timestamp for when the rate limit will reset
    reset_ts = rl.reset.timestamp()
    #calculate how many seconds to wait until reset, adding a buffer of 5 seconds
    wait = max(reset_ts - time.time(), 0) + 5
    #log a warning that we're rate limited and how long we'll wait
    log.warning(f"Rate limited. Waiting {wait:.0f}s for reset...")
    #sleep for the calculated duration
    time.sleep(wait)

#function to recursively walk a repo's file tree and collect Python files
def _get_python_files(repo, g: Github) -> list[dict]:
    """
    Recursively walk a repo's file tree and return
    a list of dicts with raw code + metadata.
    """
    #stores the collected file dicts
    collected = []

    try:
        #start with the root of the repo and get its contents
        queue = repo.get_contents("")
        #sometimes the API returns a single file instead of a list if the repo has only one item at the root, so we normalize it to a list          
        if not isinstance(queue, list):
            queue = [queue]

    except GithubException as e:
        log.warning(f"  Could not read root of {repo.full_name}: {e}")
        return []
    #use a stack to perform a depth-first traversal of the repo's file tree
    while queue and len(collected) < MAX_FILES_PER_REPO:
        while True:
            try:
                item = queue.pop(0)
                break
            except RateLimitExceededException:
                _wait_for_rate_limit(g)
        #if the item is a directory, we need to get its contents and add them to the queue to explore them later
        if item.type == "dir":
            try:
                #get the contents of the directory and add them to the queue
                children = repo.get_contents(item.path)
                
                if isinstance(children, list):
                    queue.extend(children)
                else:
                    queue.append(children)
            except (GithubException, RateLimitExceededException):
                pass
            continue

        if not item.path.endswith(".py"):
            continue

        if not (MIN_FILE_BYTES <= item.size <= MAX_FILE_BYTES):
            continue
        #try to get the raw content of the file, handling rate limits and other exceptions
        while True:
            try:
                
                raw = item.decoded_content
                break
            except RateLimitExceededException:
                _wait_for_rate_limit(g)
            except Exception as e:
                log.debug(f"  Skipping {item.path}: {e}")
                raw = None
                break

        if raw is None:
            continue
        #try to decode the raw bytes into a UTF-8 string, 
        # ignoring errors, and if that fails, skip the file
        try:
            code = raw.decode("utf-8", errors="ignore")
        except Exception:
            continue
        #append a dict with the repo and file metadata, along with the code, to our collected list
        collected.append(
            {
                "repo": repo.full_name,
                "repo_url": repo.html_url,
                "stars": repo.stargazers_count,
                "file_path": item.path,
                "file_size": item.size,
                "code": code,
            }
        )

    return collected

#main function to run all search queries and collect Python files,
#  while respecting the max repos per query and avoiding duplicates across queries
def scrape_all(g: Github) -> list[dict]:
    """
    Run all search queries and collect Python files.
    Returns a flat list of file dicts.
    """
    seen_repos: set[str] = set()
    all_files: list[dict] = []

    for query in SEARCH_QUERIES:
        log.info(f"Searching: '{query}'")
        while True:
            try:
                results = g.search_repositories(query, sort="stars", order="desc")
                break
            except RateLimitExceededException:
                _wait_for_rate_limit(g)

        count = 0
        for repo in results:
            if count >= MAX_REPOS_PER_QUERY:
                break
            if repo.full_name in seen_repos:
                continue

            seen_repos.add(repo.full_name)
            count += 1
            log.info(
                f"  [{count}/{MAX_REPOS_PER_QUERY}] Scraping {repo.full_name} *{repo.stargazers_count}"
            )

            files = _get_python_files(repo, g)
            all_files.extend(files)
            log.info(f"    -> collected {len(files)} files (total so far: {len(all_files)})")

            time.sleep(1.5)

    log.info(f"Scraping complete. Total files collected: {len(all_files)}")
    return all_files
