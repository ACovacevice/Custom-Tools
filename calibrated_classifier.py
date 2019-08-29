
# coding: utf-8

from sklearn.datasets import make_classification
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, recall_score, precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

X, y = make_classification(n_samples=500000, n_features=20, n_classes=2, weights=[.01, .99])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

def calibrated_fit(clf, X, y, n_splits=5, scoring="gini", stratify=False, calibrated_probs=True):
    
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


# Checking for consistency over time

dataframe = pd.DataFrame({
    "y": y_test, 
    "score": (1000 * best_clf.predict_proba(X_test)[:, 1]).astype(int)
})

# Randomizing some time-related variable for demonstration purposes only
dataframe["date"] = np.random.choice(range(201901, 201908, 1), size=len(frame))

def verify_consistency(df, col, target, grouper, q=200, n=20):
    
    data = df.copy()
    data["aux"] = pd.qcut(data[col], q=q, duplicates="drop")
    grouped_aux = data.groupby(["aux", grouper]).agg({target: lambda x: 100 * x.sum() / len(x)})[::-1]

    s = len(grouped_aux.index.levels[1])
    buckets = grouped_aux.index.levels[0][::-1].tolist()
    
    data = pd.DataFrame(grouped_aux[target].values.reshape(-1, s))
    
    if not data.shape == (len(buckets), s):
        return verify_consistency(df, col, target, grouper, q=(q-q//10), n=n)
    
    data["bucket"] = buckets
    data = data.sort_values("bucket").reset_index(drop=True)
    
    clf = KMeans(random_state=0, n_clusters=n)
    data["kmeans"] = clf.fit_predict(data.iloc[:, :-2], data["bucket"])
    
    for group in data["kmeans"].unique():
        idx = np.where(data["kmeans"] == group)[0]
        left = data.loc[idx, "bucket"].min().left
        right = data.loc[idx, "bucket"].max().right
        data.loc[idx, "new_bucket"] = pd.Interval(left, right)
        
    intervals = pd.IntervalIndex(sorted(data["new_bucket"].unique()))
    
    try:
        frame["bucket"] = pd.cut(frame[col], bins=intervals)
    except:
        return verify_consistency(df, col, target, grouper, q=q, n=n-1)
    
    data = frame.groupby(["bucket", grouper]).agg({"y": lambda x: 100 * x.sum() / len(x)})
    return data

def plot_consistency(df):
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    labels = df.index.levels[1]
    s = len(labels)

    for n, (i, j) in enumerate(df.index):
        if n % s == 0:
            df.loc[i].plot(use_index=False, y="y", ax=ax, label=i, marker="o", logy=False)

    ax.legend(title="Bucket", loc="upper right", bbox_to_anchor=(0.4, 0, 1, 1))        
    ax.set_ylabel("Bad Rate (%)")
    ax.set_xticklabels(labels)
    plt.show()
    
# New dataframe with the most consistent score splits over time     
df = verify_consistency(dataframe, "score", "y", "date")

# Plotting split behavior with time
plot_consistency(df)
