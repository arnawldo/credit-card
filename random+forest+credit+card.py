
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:

credit = pd.read_csv('data/creditcard.csv')


# In[3]:

credit['Time'] = credit.Time.mod(86400)


# In[4]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    credit.iloc[:, 0:-1], credit.iloc[:, -1], random_state=0)


# In[5]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:

# set parameters for grid search
param_grid = {'max_features': [2, 3, 4], 
              'max_depth': [10, 11, 12], 
              'n_estimators': [12]}
# init
grid_search = GridSearchCV(RandomForestClassifier(random_state=0, class_weight={0:.2, 1:.8}),
                           param_grid,
                           cv=3, 
                           n_jobs=-1, 
                           scoring='roc_auc')
# fit model
get_ipython().magic('time grid_search.fit(X_train, y_train);')


# In[11]:

from sklearn.metrics import classification_report
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))      
print('Classification report: \n{}'.format(classification_report(y_test, 
                                                                 grid_search.predict(X_test))))


# In[8]:

def plot_precision_recall(rf):
    from sklearn.metrics import precision_recall_curve

    # RandomForestClassifier has predict_proba
    precision_rf, recall_rf, thresholds_rf = precision_recall_curve(
        y_test, rf.predict_proba(X_test)[:, 1])
    plt.plot(precision_rf, recall_rf, label="rf")

    close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))

    plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], 'x', c='k',
    markersize=10, label="threshold 0.5 rf", fillstyle="none", mew=2)

    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.legend(loc="best")


# In[12]:

plot_precision_recall(grid_search)


# In[17]:

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, grid_search.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
# find threshold closest to zero
close_zero = np.argmin(np.abs(thresholds - 0.5))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10, 
         label="threshold zero", fillstyle="none", c='k', mew=2)
plt.legend(loc=4)


# In[ ]:



