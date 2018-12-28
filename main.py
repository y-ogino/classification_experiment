from dataset import ClassificationDataset
import evaluation as evl
import pandas as pd
import sklearn.ensemble as ens
from sklearn.tree import DecisionTreeClassifier as tree

classifiers = [
    tree(min_samples_split = 100),
    tree(min_samples_split = 40),
    tree(min_samples_split = 30),
    tree(min_samples_split = 20),
    tree(min_samples_split = 10),
    ens.AdaBoostClassifier(),
    ens.BaggingClassifier(),
    ens.ExtraTreesClassifier(),
    ens.GradientBoostingClassifier(),
    ens.RandomForestClassifier()
]

df   = pd.read_csv("../data/train+test.csv").replace({'yes': 1, 'no': 0})
data = ClassificationDataset.from_dataframe(df)

aucs = [evl.auc(data, data.train(classifier)) for classifier in classifiers]
aucs