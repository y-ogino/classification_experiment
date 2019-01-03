from sklearn import metrics

def auc(dataset, clf):
    y_proba     = clf.predict_proba(dataset.X_test)[:,1]
    return  metrics.roc_auc_score(dataset.y_test, y_proba)

def confusion_matrix(dataset, clf):
    y_pred = clf.predict(dataset.X_test)
    return metrics.confusion_matrix(dataset.y_test, y_pred)

def f1_score(dataset, clf):
    y_pred = clf.predict(dataset.X_test)
    return metrics.f1_score(dataset.y_test, y_pred, pos_label = 0)