# Argument Mining for Online Debates

NLP group project, Option 1: Application Development.

This project builds a prototype argument mining system for online debates. Given
a parent comment and a current comment, the system predicts the argumentative
role of the current comment. The goal is to help users quickly understand the
structure of a debate: what the main claims are, which replies challenge them,
which replies provide supporting reasons, and which replies are not substantive
arguments.

## Task Definition

The classifier predicts one of four labels:

| Label | Meaning |
|---|---|
| `claim` | A debatable position or main point. Root posts often fall into this class. |
| `counter_claim` | A reply that challenges, rejects, or argues against the parent comment. |
| `premise` | A reason, example, evidence, or explanation that supports a claim or counter-claim. |
| `unknown` | Non-argumentative text, such as questions, jokes, acknowledgements, or meta-comments. |

The model input has two fields:

```text
parent_text  +  current_text
```

`parent_text` can be empty when the current text is a root-level claim.

## Project Structure

```text
nlp-project/
|-- app.py                         # Streamlit demo for interactive prediction
|-- data/                          # Data schema, loaders, and preprocessing
|   |-- loaders/                    # IBM, CMV, ConvoKit, and Reddit loaders
|   `-- preprocessing/              # Text cleaning utilities
|-- src/                           # Dataset, model wrapper, and training code
|-- evaluation/                    # Custom evaluation set, scripts, metrics, plots
|   |-- custom_argument_eval.jsonl
|   |-- evaluate_custom.py
|   |-- plot_results.py
|   `-- figures/
|-- analysis/                      # Qualitative analysis artifacts
|-- models/                        # Local model files, ignored by Git
|-- requirements.txt
`-- README.md
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```


## Data Sources

The project is designed to combine formal argument data with online discussion
style data.

- IBM Debater: `ibm/argument_quality_ranking_30k`
- CMV-style discussion examples for custom evaluation
- Optional Reddit/CMV scraping utilities in `data/loaders/`
- Optional ConvoKit CMV corpus support

## Training

Training code is in:

```text
src/train.py
```

Example command:

```powershell
python -m src.train --epochs 1 --batch-size 8
```

The training pipeline supports IBM Debater data through the datasets library and
CMV-style data through local loaders. The trained model is a RoBERTa sequence
classification model for the four argument-role labels.

## Model Checkpoints

After training, the model can be used in two ways:

- `Remote`: load the trained group checkpoint used by the Streamlit app.
- `Local`: place trained model files under `models/best`.

For local usage, the expected directory is:

```text
models/best
```

Large model weights such as `model.safetensors` are not committed to GitHub.
They should be stored outside the repository or loaded through the configured
remote checkpoint.

## Run The Demo

Start the Streamlit app:

```powershell
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The demo allows users to enter:

- a parent comment
- a current comment

It returns:

- the predicted argument role
- the model confidence distribution across the four labels

## Custom Evaluation

The custom evaluation set is stored in:

```text
evaluation/custom_argument_eval.jsonl
```

It contains 48 manually reviewed examples balanced across the four labels:

| Label | Count |
|---|---:|
| `claim` | 12 |
| `counter_claim` | 12 |
| `premise` | 12 |
| `unknown` | 12 |

Run the evaluation:

```powershell
python -m evaluation.evaluate_custom
```

Generate plots:

```powershell
python -m evaluation.plot_results
```

Main evaluation artifacts:

```text
evaluation/custom_argument_eval_predictions.csv
evaluation/custom_argument_eval_metrics.json
evaluation/figures/confusion_matrix.png
evaluation/figures/label_distribution.png
```

## Current Evaluation Results

The current group model was evaluated on the 48-example custom test set.

| Metric | Score |
|---|---:|
| Accuracy | 0.6875 |
| Macro-F1 | 0.6774 |

Per-class performance:

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| `claim` | 1.0000 | 0.5000 | 0.6667 |
| `counter_claim` | 0.7500 | 1.0000 | 0.8571 |
| `premise` | 0.7143 | 0.8333 | 0.7692 |
| `unknown` | 0.4167 | 0.4167 | 0.4167 |

The strongest categories are `counter_claim` and `premise`. The main remaining
challenge is distinguishing root claims and non-argumentative replies from other
argumentative content.

## Reproducibility Notes

- Python virtual environments and local model checkpoints are ignored by Git.
- The large `model.safetensors` file should not be committed to the repository.
- Evaluation scripts write metrics, predictions, and plots under `evaluation/`.
- The custom evaluation set is small by design, so results should be interpreted
  as prototype evidence rather than a final benchmark.

## Use Of AI Tools

LLMs were used for boilerplate code assistance, debugging, writing support, and
evaluation-set drafting. The project framing, label definitions, evaluation
decisions, error interpretation, and final conclusions should be reviewed and
owned by the team.
