# Argument Mining for Online Debates

NLP group project — Option 1: Application Development.

Given a Reddit thread where people disagree, the system extracts argument structure (claims, counter-claims, premises) and surfaces what the debate is actually about.

## Project structure

```
nlp-project/
├── data/               # Data loading and preprocessing (Person 1)
├── src/                # Models and inference pipeline (Person 2)
├── evaluation/         # Custom eval harness and metrics (Person 3)
├── analysis/           # Failure mode analysis (Person 4)
├── report/             # Literature review and written report (Person 5)
└── notebooks/          # Exploration and demos (Person 6)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Reddit API credentials before using the Reddit scraper.

## Data sources

- **Change My View (CMV)** — r/changemyview threads with delta annotations as persuasion signals
- **IBM Debater** — `ibm/argument_quality_ranking_30k` via HuggingFace datasets
- **Live Reddit** — PRAW scraper for fresh threads

See `data/README.md` for details on obtaining and placing raw files.

## Use of AI tools

This project used LLMs (Claude) for boilerplate code generation and literature search. All methodology decisions, test set design, label definitions, and conclusions are the team's own.
