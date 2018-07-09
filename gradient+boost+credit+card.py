# gradient boosting machines

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')

credit = pd.read_csv('data/creditcard.csv')

credit['Time'] = credit.Time.mod(86400)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    credit.iloc[:, 0:-1], credit.iloc[:, -1], random_state=0)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


param_grid = {'max_depth': [2, 5, 10], 'learning_rate': [0.001, 0.01, 0.1]}

grid_search = GridSearchCV(GradientBoostingClassifier(random_state=0), 
                           param_grid=param_grid, 
                           n_jobs=-1, 
                           cv=5)

grid_search.fit(X_train, y_train)


print("Train set score: {:.8f}".format(grid_search.score(X_train, y_train)))
print("Test set score: {:.8f}".format(grid_search.score(X_test, y_test)))

from sklearn.metrics import roc_auc_score
rf_auc = roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1])
print("AUC for Gradient Boosting Machines: {:.3f}".format(rf_auc))

from sklearn.metrics import classification_report
print(classification_report(y_test, grid_search.predict(X_test)))


def plot_precision_recall(clf):
    # plot precision recall curve
    from sklearn.metrics import precision_recall_curve

    precision_gbc, recall_gbc, thresholds = precision_recall_curve(
        y_test, clf.predict_proba(X_test)[:, 1])
    # find threshold closest to zero
    close_zero = np.argmin(np.abs(thresholds - 0.5))

    fig, ax = plt.subplots()
    # curve
    ax.plot(precision_gbc, recall_gbc)
    # threshold
    ax.plot(precision_gbc[close_zero], recall_gbc[close_zero], 'o', markersize=10,
    label="threshold zero 0.5", fillstyle="none", c='k', mew=2)
    # aes
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.legend(loc="best")

    plt.show()

def plot_feature_importances(model):
    n_features = len(credit.columns) - 1
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), credit.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")


