from src import predict_debate
from data import load_convokit_cmv, clean_debates

# Load a real CMV thread from the test split
debates = clean_debates(load_convokit_cmv("test"))

# Pick the first thread with enough arguments to be interesting
debate = next(d for d in debates if len(d.arguments) >= 5)

print(f"Thread: {debate.title}")
print("=" * 70)

labeled = predict_debate(debate)

for arg in labeled.arguments:
    indent = "    " if arg.parent_id else ""
    label = arg.arg_type.upper().replace("_", " ")
    print(f"{indent}[{label}] {arg.text[:120].strip()}...")
    print()
