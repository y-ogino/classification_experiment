from dataset import ClassificationDataset
from evaluation import auc, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt

df   = pd.read_csv("../data/train+test.csv").replace({'yes': 1, 'no': 0})
data = ClassificationDataset.from_dataframe(df)

classifier = DecisionTreeClassifier(min_samples_split = 20)
clf        = data.train(classifier)

auc(data, clf)
confusion_matrix(data, clf)
f1_score(data, clf)