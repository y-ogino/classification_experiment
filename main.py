from pprint import pprint
from dataset import LabeledDataset
from classifiers import classifiers
import evaluation as evl
import pandas as pd

def test(data, classifier):
    clf = data.train(classifier)
    auc = evl.auc(data, clf)
    f1  = evl.f1_score(data, clf)
    return {"AUC": auc, "F-measure": f1}

df     = pd.read_csv("../data/train+test.csv").replace({'yes': 1, 'no': 0})
data   = LabeledDataset.from_dataframe(df)
scores = [test(data, classifier) for classifier in classifiers]
pprint(scores)