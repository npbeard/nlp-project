# User Manual: Argument Role Classifier

## Overview

This application classifies the argumentative role of a comment in an online
debate. Given a parent comment and a current comment, the system predicts
whether the current comment is a `claim`, `counter_claim`, `premise`, or
`unknown`.

The application is intended as a proof-of-concept tool for exploring discussion
structure in online debates. It is not a production moderation system.

## Requirements

- Python 3.9 or later
- Internet connection for loading the remote model checkpoint
- Project dependencies installed from `requirements.txt`

## Installation

From the project root, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

## Running The Application

Start the Streamlit demo:

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How To Use The Demo

1. Select an example from the dropdown, or replace the text with your own.
2. Enter the previous comment in the `Parent text` box.
3. Enter the comment you want to classify in the `Current text` box.
4. Click `Classify`.
5. Read the predicted label and confidence scores.

If the comment is a root-level post with no parent, leave `Parent text` empty
and only fill in `Current text`.

## Label Guide

| Label | Meaning |
|---|---|
| `claim` | A debatable position or main assertion. |
| `counter_claim` | A reply that challenges or argues against the parent comment. |
| `premise` | A reason, example, or evidence supporting a claim or counter-claim. |
| `unknown` | A question, acknowledgement, vague reply, joke, or off-topic comment. |

## Example Inputs

### Example 1: Counter-Claim

Parent text:

```text
Universities should allow students to use generative AI tools in coursework.
```

Current text:

```text
AI tools should be banned from graded assignments because they make authorship impossible to verify.
```

Expected interpretation:

```text
counter_claim
```

### Example 2: Premise

Parent text:

```text
Cities should make public transport free for residents.
```

Current text:

```text
Fare collection systems are expensive to maintain, so removing fares can reduce administrative costs.
```

Expected interpretation:

```text
premise
```

### Example 3: Unknown

Parent text:

```text
Detection systems have produced false positives against students who did not use AI.
```

Current text:

```text
Do you have evidence for that claim?
```

Expected interpretation:

```text
unknown
```

## Model Options

The sidebar provides two model-loading options:

- `Remote`: loads the trained group checkpoint used by the demo.
- `Local`: loads a local checkpoint from `models/best`.

The remote option is recommended for most users. The local option is useful only
if the trained model files have already been placed under `models/best`.

## Evaluation Artifacts

The custom evaluation files are stored in the `evaluation/` folder:

```text
evaluation/custom_argument_eval.jsonl
evaluation/custom_argument_eval_predictions.csv
evaluation/custom_argument_eval_metrics.json
evaluation/figures/confusion_matrix.png
evaluation/figures/label_distribution.png
```

To reproduce the evaluation:

```powershell
python -m evaluation.evaluate_custom
python -m evaluation.plot_results
```

## Known Limitations

- Root-level claims without parent context are sometimes predicted as `unknown`.
- Clarifying questions can be over-interpreted as argumentative content.
- Short acknowledgements or vague replies may be confused with `premise` or
  `counter_claim`.
- Informal forum language is harder than clean debate-style examples.

These limitations are discussed in the failure mode analysis.

## Troubleshooting

If the app cannot load the model, check that:

- the internet connection is available when using `Remote`
- local model files exist under `models/best` when using `Local`
- dependencies were installed with `pip install -r requirements.txt`

If Streamlit does not start, make sure the virtual environment is activated and
run:

```powershell
streamlit run app.py
```
