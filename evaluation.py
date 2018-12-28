from sklearn import metrics

def auc(dataset, clf):
    y_proba     = clf.predict_proba(dataset.X_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(dataset.y_test, y_proba)
    return  metrics.auc(fpr, tpr)

def confusion_matrix(dataset, clf):
    y_pred = clf.predict(dataset.X_test)
    return metrics.confusion_matrix(dataset.y_test, y_pred)

def f1_score(dataset, clf):
    y_pred = clf.predict(dataset.X_test)
    return metrics.f1_score(dataset.y_test, y_pred)