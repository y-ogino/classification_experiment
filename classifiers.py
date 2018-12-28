from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier

base = DecisionTreeClassifier(max_depth = 5, class_weight = "balanced")

classifiers = [
    base,
    DecisionTreeClassifier(min_samples_split = 100, class_weight = "balanced"),
    AdaBoostClassifier(n_estimators = 50, base_estimator = base),
    BaggingClassifier(n_estimators = 50, base_estimator = base),
    GradientBoostingClassifier(n_estimators = 50),
    RandomForestClassifier(n_estimators = 50, max_depth = 5, class_weight = "balanced")
]
