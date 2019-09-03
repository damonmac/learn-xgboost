#!/usr/bin/python

import sys
sys.path.append("./")
from prepare import preprocess
from sklearn.metrics import accuracy_score, average_precision_score
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = preprocess("../data/vector3")

params = {'n_estimators': 1000,
          'n_jobs': 4,
          'eta': 0.1,
          'max_depth': 6,
          'gamma': 0,
          'subsample': 1,
          'colsample_bytree': 1,
          'objective': 'binary:logistic'}

clf = XGBClassifier(**params)

eval_set = [(X_train, y_train),(X_test, y_test)]
clf.fit(X_train, y_train, eval_metric=["logloss"], eval_set=eval_set, verbose=100)

predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

y_score = clf.predict_proba(X_test)[:,1]
average_precision = average_precision_score(y_test, y_score)
print("Average precision score: %.2f%%" % (average_precision * 100.0))