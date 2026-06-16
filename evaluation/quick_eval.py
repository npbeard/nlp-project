from src import predict_debate
from data import load_ibm, clean_debates
from sklearn.metrics import classification_report

debates = clean_debates(load_ibm("test"))[:60]

all_true, all_pred = [], []
for debate in debates:
    labeled = predict_debate(debate)
    for orig, pred in zip(debate.arguments, labeled.arguments):
        all_true.append(orig.arg_type)
        all_pred.append(pred.arg_type)

print(classification_report(all_true, all_pred))
