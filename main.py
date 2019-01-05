from dataset import LabeledDataset
from classifiers import classifiers, classifier_names
import evaluation as evl
import pandas as pd

def test(data, classifier):
    clf = data.train(classifier)
    auc = evl.auc(data, clf)
    f1  = evl.f1_score(data, clf)
    return {"AUC": auc, "F-measure": f1}

if __name__ == "__main__":
    df     = pd.read_csv("../data/train+test.csv").replace({'yes': 1, 'no': 0})
    data   = LabeledDataset.from_dataframe(df)
    scores = [test(data, classifier) for classifier in classifiers]

    scores_df = pd.DataFrame(data = scores, index = classifier_names)
    print(scores_df)