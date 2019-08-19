#!/usr/bin/python

import sys
sys.path.append("./")
from prepare import preprocess
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = preprocess("./data/vector3")

params = {'n_estimators': 700, 
          'max_depth': 7, 
          'gamma': 0.95, 
          'subsample': 0.95, 
          }

clf = XGBClassifier(**params)

eval_set = [(X_train, y_train),(X_test, y_test)]
clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric=["logloss"], eval_set=eval_set, verbose=True)

predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))