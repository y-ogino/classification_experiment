from dataset import ClassificationDataset
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt

df   = pd.read_csv("../data/train+test.csv").replace({'yes': 1, 'no': 0})
data = ClassificationDataset.from_dataframe(df)

classifier  = DecisionTreeClassifier(min_samples_split = 20)
fpr, tpr, _ = data.evaluate(classifier)

plt.plot(fpr, tpr)
plt.show()