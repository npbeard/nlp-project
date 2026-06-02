# Data module

Owns all data loading, cleaning, and the unified schema. Everything downstream imports from here.

## Unified schema

```python
from data import Argument, Debate

debate = Debate(id="abc", title="CMV: cats > dogs", source="cmv")
arg   = Argument(id="1", text="Cats are independent.", arg_type="premise")
```

`arg_type` must be one of: `claim`, `counter_claim`, `premise`, `unknown`.

## Sources

### CMV (offline)

Download the Tan et al. 2016 dataset:
```bash
mkdir -p data/raw/cmv
curl -O https://chenhaot.com/data/cmv/cmv.tar.bz2
tar -xjf cmv.tar.bz2 -C data/raw/cmv/
```

Then:
```python
from data import load_cmv, clean_debates
debates = clean_debates(load_cmv("train"))
```

### IBM Debater (auto-download)

```python
from data import load_ibm
debates = load_ibm("train")  # downloads via HuggingFace datasets on first run
```

### Live Reddit scraper

Add credentials to `.env` (copy from `.env.example`):
```
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=nlp-project/1.0 by your_username
```

Then:
```python
from data import scrape_cmv, clean_debates
debates = clean_debates(scrape_cmv(limit=100, sort="top", time_filter="month"))
```

## Preprocessing

`clean_debates()` applies Reddit-specific noise removal (quoted text, URLs,
edit notes, user mentions) and drops arguments shorter than 5 tokens. Call it
on any list of `Debate` objects before passing them downstream.
