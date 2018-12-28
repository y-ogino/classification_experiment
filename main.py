from dataset import LabeledDataset
from classifiers import classifiers
import evaluation as evl
import pandas as pd

def test(data, classifier):
    clf = data.train(classifier)
    return evl.auc(data, clf)

df   = pd.read_csv("../data/train+test.csv").replace({'yes': 1, 'no': 0})
data = LabeledDataset.from_dataframe(df)
aucs = [test(data, classifier) for classifier in classifiers]
aucs