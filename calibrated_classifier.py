
# coding: utf-8

from sklearn.datasets import make_classification
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, recall_score, precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import numpy as np

X, y = make_classification(n_samples=500000, n_features=20, n_classes=2, weights=[.01, .99])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

def calibrated_fit(clf, X, y, n_splits=5, scoring="gini", threshold=.5, stratify=False, calibrated_probs=True):
    
    assert hasattr(clf, "predict_proba"), "Classifier has no attribute predict_proba."

    trained_models, scores = [], {"train": [], "test": []}
    
    if calibrated_probs:
        clf = CalibratedClassifierCV(clf, cv=3)
    
    if stratify:
        kfold = StratifiedKFold(n_splits, shuffle=True, random_state=0)
    else:    
        kfold = KFold(n_splits, shuffle=True, random_state=0)

    for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):

        X_train, y_train = X[train_index], y[train_index]
        X_test,  y_test =  X[test_index],  y[test_index]

        clf.fit(X_train, y_train)
        trained_models.append(clf)

        y_pred_train = clf.predict_proba(X_train)[:, 1] 
        train_scores = []
        
        for threshold in (.5, .6, .7, .8, .9):
        
            if scoring == "accuracy":
                train_score = accuracy_score(y_train, y_pred_train > threshold)
            elif scoring == "gini":
                train_score = 2 * roc_auc_score(y_train, y_pred_train > threshold) - 1
            elif scoring == "f1": 
                train_score = f1_score(y_train, y_pred_train > threshold)
            elif scoring == "precision":
                train_score = precision_score(y_train, y_pred_train > threshold)
            elif scoring == "recall":
                train_score = recall_score(y_train, y_pred_train > threshold)
            
            train_scores.append(train_score)
        
        scores["train"].append(train_scores)
        
        y_pred_test = clf.predict_proba(X_test)[:, 1] 
        test_scores = []
        
        for threshold in (.5, .6, .7, .8, .9):
            
            if scoring == "accuracy":
                test_score = accuracy_score(y_test, y_pred_test > threshold)        
            elif scoring == "gini":
                test_score = 2 * roc_auc_score(y_test, y_pred_test > threshold) - 1
            elif scoring == "f1": 
                test_score = f1_score(y_test, y_pred_test > threshold)
            elif scoring == "precision":
                test_score = precision_score(y_test, y_pred_test > threshold)
            elif scoring == "recall":
                test_score = recall_score(y_test, y_pred_test > threshold)
            
            test_scores.append(test_score)
        
        scores["test"].append(test_scores)            
            
        print(" # Fold number %i:" % (fold + 1))
        print("Train scores: ", train_scores)
        print("Test  scores: ", test_scores)
    
    return trained_models, scores

def plot_calibration_curve(clf, X_test, y_test, y, name):

    fig = plt.figure(1, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for clf, name in [(clf, name)]:
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            
        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)    

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % (name, clf_score))
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="upper left")
        ax1.set_title('Calibration plots  (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()
        
# Fitting a classifier with calibrated probabilities
estimator = LogisticRegression(solver="lbfgs")
clf, scores = calibrated_fit(estimator, X_train, y_train, scoring="gini", stratify=True, calibrated_probs=True)

# Printing train scores
print(np.array(scores["train"]).mean(axis=1))

# Printing test scores
print(np.array(scores["test"]).mean(axis=1))

# Selecting the top performer
best_clf = clf[np.array(scores["test"]).mean(axis=1).argmax()]

# Plotting the calibration curve
plot_calibration_curve(best_clf, X_test, y_test, y, estimator.__class__)
plt.show()
