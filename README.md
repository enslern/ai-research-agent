# AI Research Agent

An autonomous research agent that takes a high-level goal, breaks it into focused sub-questions, searches the web, scrapes and summarises sources, scores them for quality, and synthesises everything into a structured markdown report.

---

## Architecture

```
ai-research-agent/
│
├── agent/
│   ├── planner.py       # Decomposes a goal into research tasks (Groq LLM)
│   ├── researcher.py    # Orchestrates the per-task research loop
│   ├── summariser.py    # Condenses raw webpage content (Groq LLM)
│   ├── evaluator.py     # Scores sources for relevance + credibility (Groq LLM)
│   ├── confidence.py    # Weighted confidence scoring formula
│   └── reporter.py      # Synthesises findings into a markdown report
│
├── browser/
│   ├── search.py        # Web search via Tavily API
│   └── scrapper.py      # Page fetching via Playwright + Readability
│
├── memory/
│   └── vector_store.py  # FAISS-backed semantic memory (sentence-transformers)
│
├── storage/
│   └── state_manager.py # JSON state persistence with numpy-safe serialisation
│
├── tests/               # Full pytest suite (mocked — no API calls needed)
├── reports/             # Generated markdown reports land here
├── main.py              # Entry point
├── requirements.txt
├── pytest.ini
└── .env.example
```

### How a research run works

```
Goal
 │
 ▼
[Planner]  →  5 sub-tasks
 │
 ▼  for each task:
[Memory]   →  hit?  ──yes──▶  use cached results
               │
              no
               ▼
[Search]   →  URLs
               ▼
[Scraper]  →  raw HTML  →  structured text
               ▼
[Summariser] → 150-word summary
               ▼
[Evaluator]  → relevance / credibility / authority scores
               ▼
[Confidence] → single 0–1 score
               ▼
[Filter]   →  relevance > 0.6 AND credibility > 0.5?
               │
              yes
               ▼
[Memory.add] + [State.save]
 │
 ▼
[Reporter]  →  LLM synthesis  →  reports/report_<timestamp>.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/yourname/ai-research-agent.git
cd ai-research-agent
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

| Key | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — free tier available |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) — free tier available |

### 3. Run

```bash
# Default goal (hardcoded in main.py)
python main.py

# Custom goal
python main.py "best practices for distributed systems interviews"

# Start fresh (ignore saved state)
python main.py --fresh "your new goal here"
```

The report is saved to `reports/report_<timestamp>.md`.

---

## Configuration

| Constant | File | Default | Description |
|---|---|---|---|
| `GOAL` | `main.py` | `"how to solve 3000 leetcode..."` | Default research goal |
| `_MIN_RELEVANCE` | `researcher.py` | `0.60` | Minimum relevance to keep a source |
| `_MIN_CREDIBILITY` | `researcher.py` | `0.50` | Minimum credibility to keep a source |
| `_MAX_PARAGRAPH_CHARS` | `summariser.py` | `6000` | Truncation limit for scraped content |
| `_PAGE_TIMEOUT_MS` | `scrapper.py` | `15000` | Playwright page load timeout |
| `_MAX_RETRIES` | `planner.py` | `3` | LLM retry attempts before raising |

---

## Confidence scoring formula

```
confidence = 0.40 × relevance
           + 0.35 × credibility
           + 0.15 × authority        (rule-based domain heuristic)
           + 0.10 × agreement        (grows 0→1 as corroborating sources accumulate, saturates at 5)
```

`agreement` starts at **0.0** (not 0.5) so the first result in any task is scored conservatively on its own merits. Confidence grows as evidence accumulates.

---

## Running tests

No API keys required — all external calls are mocked.

```bash
pytest
```

Run a single test module:

```bash
pytest tests/test_vector_store.py -v
```

Run with coverage report only (no HTML):

```bash
pytest --no-header --cov=agent --cov=memory --cov=storage --cov-report=term-missing
```

### What's tested

| Module | Test file | Key cases |
|---|---|---|
| `memory/vector_store.py` | `test_vector_store.py` | Nested-def bug regression, add/search, serialisation round-trip |
| `agent/confidence.py` | `test_confidence.py` | First-result inflation bug regression, saturation at 5 results |
| `agent/evaluator.py` | `test_evaluator.py` | Authority blending, fallback on API failure, malformed JSON |
| `agent/planner.py` | `test_planner.py` | Markdown fence stripping, retry logic, max-retry raise |
| `agent/summariser.py` | `test_summariser.py` | Content truncation, fallback on failure |
| `agent/reporter.py` | `test_reporter.py` | File creation, goal in output, LLM fallback |
| `storage/state_manager.py` | `test_state_manager.py` | Numpy serialisation bug, round-trip, corrupt JSON handling |

---


## Tech stack

| Layer | Library |
|---|---|
| LLM inference | [Groq](https://groq.com) — `llama-3.3-70b-versatile` |
| Web search | [Tavily](https://tavily.com) |
| Web scraping | [Playwright](https://playwright.dev/python/) + [readability-lxml](https://github.com/buriy/python-readability) + [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) |
| Vector memory | [FAISS](https://github.com/facebookresearch/faiss) + [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| State persistence | `json` + custom numpy encoder |
| Testing | [pytest](https://pytest.org) + [pytest-cov](https://pytest-cov.readthedocs.io) |

---

## Project status

This is a working prototype. Known limitations and possible next steps:

- **Playwright in CI**: headless Chromium adds significant setup weight. Swap for `httpx` + `trafilatura` for environments where browser install is not feasible.
- **FAISS is in-memory only**: the index is rebuilt from saved metadata on each resume. For large-scale use, swap for a persistent vector DB (Chroma, Qdrant).
- **Single-threaded scraping**: URLs are fetched sequentially. Parallelising with `concurrent.futures.ThreadPoolExecutor` would speed up large task lists significantly.
- **No deduplication**: the same URL can be scraped twice across tasks. A URL-level bloom filter or set in `VectorStore` would prevent this.
