import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

def split(df):
    m   = int(len(df.index) / 2)
    df1 = df.iloc[:m]
    df2 = df.iloc[m:]
    return (df1, df2)

def get_csv(filename):
    df = pd.read_csv(filename).replace({'yes': 1, 'no': 0})
    X  = pd.get_dummies(df.drop(columns = 'y'))
    y  = df['y']
    return (split(X), split(y))

def evaluate(X, y, X_test, y_test, classifier):
    clf     = classifier.fit(X, y)
    y_proba = clf.predict_proba(X_test)[:,1]

    return metrics.roc_curve(y_test, y_proba)


((X, X_test), (y, y_test)) = get_csv("../data/train+test.csv")

classifier = DecisionTreeClassifier(min_samples_split = 20)
fpr, tpr, thresholds = metrics.roc_curve(X, y, X_test, y_test, classifier)

plt.plot(fpr, tpr)
plt.show()