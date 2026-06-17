# Failure Mode Analysis: Custom Argument Role Evaluation

## Evaluation Setup

We evaluated the trained argument role classifier on a custom 48-example test
set. Each example contains:

- `parent_text`: the previous comment in the discussion, empty for root claims
- `current_text`: the comment being classified
- `gold_label`: the manually assigned argument role
- `rationale`: a short explanation for the label
- `probe_type`: the linguistic or discourse pattern being tested

The test set is balanced across the four labels:

| Label | Count |
|---|---:|
| claim | 12 |
| counter_claim | 12 |
| premise | 12 |
| unknown | 12 |

Most examples are controlled debate probes written to test clear argument-role
distinctions. Eight additional examples use more informal forum-style language
about vibe coding and creativity, drawn from real online discussion.

The evaluated model is the trained RoBERTa classifier used by the group demo.


## Quantitative Results

| Metric | Score |
|---|---:|
| Accuracy | 0.6875 |
| Macro-F1 | 0.6774 |

Per-class scores:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| claim | 1.0000 | 0.5000 | 0.6667 | 12 |
| counter_claim | 0.7500 | 1.0000 | 0.8571 | 12 |
| premise | 0.7143 | 0.8333 | 0.7692 | 12 |
| unknown | 0.4167 | 0.4167 | 0.4167 | 12 |

Confusion matrix (rows = gold label, columns = predicted label):

| Gold \ Predicted | claim | counter_claim | premise | unknown |
|---|---:|---:|---:|---:|
| claim | 6 | 0 | 0 | 6 |
| counter_claim | 0 | 12 | 0 | 0 |
| premise | 0 | 1 | 10 | 1 |
| unknown | 0 | 3 | 4 | 5 |

## Main Findings

The model performs best on `counter_claim` (F1 0.86) and `premise` (F1 0.77).
This is encouraging for the debate-mining use case because these two labels
capture the most common reply-level argument moves in online discussions.

The weakest label is `unknown` (F1 0.42): many questions, acknowledgements,
hedging replies, and vague comments are still interpreted as argumentative
content. Root and reply-level `claim` detection is the second weakest area
(recall 0.50): the model misses half of all genuine claims, almost always
defaulting to `unknown` when the claim lacks a clear argumentative parent or
reply context.

## Summary of Documented Errors

| Example | Gold | Predicted | Probe type |
|---|---|---|---|
| eval_001 | claim | unknown | root_claim |
| eval_004 | unknown | premise | evidence_request |
| eval_005 | claim | unknown | root_claim |
| eval_016 | unknown | premise | agreement_only |
| eval_020 | unknown | counter_claim | vague_reply |
| eval_025 | claim | unknown | root_claim |
| eval_028 | unknown | counter_claim | question |
| eval_032 | unknown | premise | uncertain_reply |
| eval_037 | claim | unknown | root_claim |
| eval_039 | premise | counter_claim | example_support |
| eval_041 | claim | unknown | root_claim |
| eval_042 | claim | unknown | reply_claim |
| eval_046 | premise | unknown | supporting_explanation |
| eval_047 | unknown | premise | clarification_question |

## Failure Modes

### 1. Root Claims Without Parent Context Classified as Unknown

Six of twelve gold `claim` examples were predicted as `unknown`. Five of these
were root-level claims with an empty `parent_text`, and one was a reply-level
claim introduced within an ongoing thread.

Representative example:

| Field | Value |
|---|---|
| Example | `eval_001` |
| Parent | *(empty)* |
| Current | Universities should allow students to use generative AI tools in coursework. |
| Gold | claim |
| Predicted | unknown |

Other misclassified root-level claims: `eval_005`, `eval_025`, `eval_037`, `eval_041`.

**Hypothesis:** The training data (IBM Debater) frames every argument as a
reply to a topic motion. When the parent field is empty the model receives no
contrastive signal and may interpret a standalone sentence as low-information
content rather than a debate-opening position. Root claims require a different
representational cue — the absence of a parent — but the model was not
explicitly trained to treat that absence as a signal.

---

### 2. Reply-Level Claims Confused with Unknown

The model predicts `claim` almost exclusively at root level. When a genuine
claim appears as a reply inside a thread it is often misclassified as `unknown`.

Representative example:

| Field | Value |
|---|---|
| Example | `eval_042` |
| Parent | I think we will also see more and more designs of these apps/experiences become increasingly similar. |
| Current | Creativity shouldn't be sought in coming up with unique UIs, but in the functionality. |
| Gold | claim |
| Predicted | unknown |

**Hypothesis:** The model has learned to associate `claim` with the absence of
a parent. A comment that introduces a new debatable position mid-thread — a
reply claim — lacks that surface cue. Without it, the model falls back to
`unknown`, which is the most common non-argumentative label in online
discussion data.

---

### 3. Clarifying and Evidence-Request Questions Over-Interpreted as Arguments

Some questions that request evidence or clarification were classified as
`premise` or `counter_claim` even though they do not themselves advance an
argument.

Representative examples:

| Field | Value |
|---|---|
| Example | `eval_004` |
| Parent | Detection systems have produced false positives against students who did not use AI. |
| Current | Do you have evidence for that claim? |
| Gold | unknown |
| Predicted | premise |

| Field | Value |
|---|---|
| Example | `eval_028` |
| Parent | Several pilot programs reported stable productivity and higher employee satisfaction. |
| Current | Which pilot programs are you referring to? |
| Gold | unknown |
| Predicted | counter_claim |

**Hypothesis:** Both questions are topically close to their parent and share
surface patterns with argumentative replies (they respond to a specific
factual claim). The model appears to use topical relevance and reactive stance
as proxies for argumentative role, without reliably distinguishing between
producing evidence and requesting it. The question mark is insufficient as a
signal because rhetorical questions and leading questions do appear in genuine
`counter_claim` and `premise` turns.

---

### 4. Acknowledgements and Agreement Replies Treated as Premises

Short agreement-like replies with no new content are sometimes classified as
`premise`.

Representative examples:

| Field | Value |
|---|---|
| Example | `eval_016` |
| Parent | They may reduce visible clothing differences, but they do not address deeper income inequality. |
| Current | Exactly, that is what I meant. |
| Gold | unknown |
| Predicted | premise |

| Field | Value |
|---|---|
| Example | `eval_032` |
| Parent | A strict ban would also block students from using accessibility tools and translation apps. |
| Current | Maybe, but I am not sure how that would work in practice. |
| Gold | unknown |
| Predicted | premise |

**Hypothesis:** The model has learned that supportive and responsive replies
are typically premises, but it does not require the current text to contain a
new reason, example, or piece of evidence. Pure acknowledgements ("Exactly")
and hedging responses ("Maybe, but I am not sure") satisfy the surface
condition of replying supportively without actually contributing argumentative
substance.

---

### 5. Hedging and Vague Qualifications Mistaken for Counter-Claims

Short uncertain replies that qualify or partially push back were sometimes
predicted as `counter_claim`.

Representative example:

| Field | Value |
|---|---|
| Example | `eval_020` |
| Parent | Building new nuclear plants takes too long to help with urgent emissions targets. |
| Current | That depends on the country and the project. |
| Gold | unknown |
| Predicted | counter_claim |

**Hypothesis:** Hedging phrases such as "that depends" carry a weak
contrastive signal that the model treats as disagreement. Without a threshold
for argumentative strength, the classifier cannot distinguish a genuine
counter-claim — one that asserts an opposing position — from a vague
qualification that leaves the original claim largely intact.

---

### 6. Evidence for a Claim Confused with a Counter-Claim

One premise was predicted as `counter_claim` when the supporting example could
be read as introducing a new dimension to the debate.

Representative example:

| Field | Value |
|---|---|
| Example | `eval_039` |
| Parent | Online anonymity is necessary for free expression. |
| Current | Whistleblowers may only speak publicly if their real identity is protected. |
| Gold | premise |
| Predicted | counter_claim |

**Hypothesis:** The current comment introduces a concrete case (whistleblower
protection) that the model may read as a new argumentative move rather than
evidence for the parent claim. When a premise is specific and concrete it can
resemble a counter-claim if the model does not correctly anchor the supporting
relation. The error may also reflect ambiguity in the label itself: one could
argue the comment shifts the frame slightly while still supporting anonymity.

---

### 7. Informal and Speculative Forum Language Is Harder to Classify

The vibe coding examples use longer, more speculative discussion language drawn
from actual online forums. Premises in this subset that develop a point
indirectly were more likely to be misclassified.

Representative example:

| Field | Value |
|---|---|
| Example | `eval_046` |
| Parent | I think that is actually the opposite: vibe coding drastically reduces the level of expertise needed to prototype and implement ideas. The creativity will come from people using the tools as they try new and varied things. |
| Current | Right now people are still figuring out how to use these tools, but the more people get comfortable, the more they will figure out how to guide towards unique and interesting results. |
| Gold | premise |
| Predicted | unknown |

Also affected: `eval_047` — a clarification question in the same thread
predicted as `premise`.

**Hypothesis:** The training data (IBM Debater) consists of clean, argument-
style text written for a formal debate corpus. Informal forum replies are
longer, hedge more frequently, and develop a supporting point through
implication rather than explicit statement. The model may not have enough
exposure to this register to distinguish a speculative supporting explanation
from a non-argumentative comment.

---

## Implications

The classifier is usable as a prototype for surfacing argumentative structure
in online debates. It is strongest when a reply clearly attacks or supports a
parent comment using language close to formal argument style. The main
limitation is boundary detection: the model struggles to separate genuine
argument components from conversational moves that look superficially
argumentative (questions, hedges, acknowledgements).

Two structural weaknesses stand out. First, the model conflates the absence of
a parent with the absence of argumentative content, making root and reply-level
claims the hardest category. Second, the `unknown` class is under-served: the
model has learned that topical relevance to a parent implies an argumentative
role, and it does not reliably fall back to `unknown` when the current comment
fails to contribute substance.

Practical improvements would include augmenting the training set with more
`unknown` examples covering questions, acknowledgements, hedges, and meta-
comments; experimenting with an explicit signal for empty parent context; and
using informal forum data (Reddit, CMV) alongside IBM Debater to reduce the
register gap between training and real online discussions.
