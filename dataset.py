import pandas as pd

def split(df):
    m   = int(len(df.index) / 2)
    df1 = df.iloc[:m]
    df2 = df.iloc[m:]
    return (df1, df2)

class LabeledDataset:
    def __init__(self, X, y, X_test, y_test):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test

    @classmethod
    def from_dataframe(cls, df):
        X  = pd.get_dummies(df.drop(columns = 'y'))
        y  = df['y']
        X, X_test = split(X)
        y, y_test = split(y)
        return cls(X, y, X_test, y_test)

    def train(self, classifier):
        return classifier.fit(self.X, self.y)