from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast


REMOTE_CHECKPOINT = "Pengchong1113/argument-role-classifier"
LOCAL_CHECKPOINT = "models/best"
LABELS = ["claim", "counter_claim", "premise", "unknown"]
LABEL_DISPLAY = {
    "claim": "Claim",
    "counter_claim": "Counter-claim",
    "premise": "Premise",
    "unknown": "Unknown",
}
LABEL_HELP = {
    "claim": "A debatable position or main assertion.",
    "counter_claim": "A reply that challenges the parent comment.",
    "premise": "A reason, example, or evidence used to support an argument.",
    "unknown": "A question, acknowledgement, vague reply, or off-topic comment.",
}


EXAMPLES = {
    "AI in education": {
        "parent": "Universities should allow students to use generative AI tools in coursework.",
        "current": "AI tools should be banned from graded assignments because they make authorship impossible to verify.",
    },
    "Remote work": {
        "parent": "Remote work should remain the default for knowledge workers.",
        "current": "That ignores how much junior employees learn from being around experienced coworkers in person.",
    },
    "Public transport": {
        "parent": "Cities should make public transport free for residents.",
        "current": "Fare collection systems are expensive to maintain, so removing fares can reduce administrative costs.",
    },
    "No parent": {
        "parent": "",
        "current": "Online anonymity is necessary for free expression.",
    },
}


st.set_page_config(
    page_title="Argument Role Classifier",
    page_icon="",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    .label-box {
        border: 1px solid #d9dee8;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        background: #f8fafc;
    }
    .label-title {
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .small-muted {
        color: #5f6b7a;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def load_model(checkpoint: str):
    path = Path(checkpoint)
    model_source = str(path) if path.exists() else checkpoint
    if path.exists() and not (path / "model.safetensors").exists() and not (
        path / "pytorch_model.bin"
    ).exists():
        raise FileNotFoundError(
            f"No model weights found in {path}. Expected model.safetensors "
            "or pytorch_model.bin."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_source)
    model = RobertaForSequenceClassification.from_pretrained(model_source)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_with_scores(
    parent_text: str,
    current_text: str,
    checkpoint: str,
) -> tuple[str, float, dict[str, float]]:
    tokenizer, model, device = load_model(checkpoint)
    if parent_text.strip():
        encoded = tokenizer(
            parent_text,
            current_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length",
        )
    else:
        encoded = tokenizer(
            current_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length",
        )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits.squeeze(0)
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()

    scores = dict(zip(LABELS, probabilities))
    label = max(scores, key=scores.get)
    return label, scores[label], scores


def render_label_reference() -> None:
    cols = st.columns(4)
    for col, label in zip(cols, LABELS):
        with col:
            st.markdown(
                f"""
                <div class="label-box">
                  <div class="label-title">{LABEL_DISPLAY[label]}</div>
                  <div class="small-muted">{LABEL_HELP[label]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


st.title("Argument Role Classifier")

with st.sidebar:
    st.header("Model")
    checkpoint_options = ["Remote", "Local"] if REMOTE_CHECKPOINT else ["Local"]
    model_location = st.radio(
        "Checkpoint source",
        checkpoint_options,
        index=0,
    )
    checkpoint = (
        REMOTE_CHECKPOINT if model_location == "Remote" else LOCAL_CHECKPOINT
    )
    if model_location == "Local" and not Path(LOCAL_CHECKPOINT).exists():
        st.info("Place local model files under models/best, or switch to Remote.")

render_label_reference()

st.divider()

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    selected = st.selectbox("Example", list(EXAMPLES.keys()))
    example = EXAMPLES[selected]

    parent_text = st.text_area(
        "Parent text",
        value=example["parent"],
        height=160,
        placeholder="Optional. Paste the comment being replied to.",
    )
    current_text = st.text_area(
        "Current text",
        value=example["current"],
        height=180,
        placeholder="Paste the comment to classify.",
    )
    run_prediction = st.button("Classify", type="primary", use_container_width=True)

with right:
    if run_prediction:
        if not current_text.strip():
            st.warning("Current text is required.")
        else:
            try:
                label, confidence, scores = predict_with_scores(
                    parent_text=parent_text,
                    current_text=current_text,
                    checkpoint=checkpoint,
                )
                st.metric("Predicted label", LABEL_DISPLAY[label])
                st.metric("Confidence", f"{confidence:.1%}")

                score_rows = pd.DataFrame(
                    {
                        "label": [LABEL_DISPLAY[label] for label in LABELS],
                        "probability": [scores[label] for label in LABELS],
                    }
                )
                st.bar_chart(
                    score_rows,
                    x="label",
                    y="probability",
                    height=280,
                )

                st.dataframe(
                    score_rows.assign(
                        probability=score_rows["probability"].map(
                            lambda value: f"{value:.3f}"
                        )
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(str(exc))
    else:
        st.info("Enter text and click Classify.")
