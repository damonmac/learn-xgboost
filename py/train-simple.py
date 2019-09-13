#!/usr/bin/python

import sys
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score

# load the feature names
with open("data/features.txt", "r") as f:
  feature_names = f.read().split("\n")[:-1]

train = xgb.DMatrix('data/javaVector_train.libsvm', feature_names=feature_names)
test = xgb.DMatrix('data/javaVector_eval.libsvm', feature_names=feature_names)

eval_metrics = ['error', 'logloss']
num_boost_rounds = 1000

params = {'n_jobs': 4,
          'eta': 0.1,
          'max_depth': 4,
          'gamma': 0,
          'subsample': 1,
          'colsample_bytree': 1,
          'eval_metric': eval_metrics,
          'objective': 'binary:logistic'}

results = {}
watchlist = [(train, 'train'), (test, 'test')]
clf = xgb.train(params, train, num_boost_rounds, watchlist, evals_result = results, verbose_eval=100)

print("Calculating predictions")
predictions = clf.predict(test)
predict_labels = np.array([1 if x >= 0.6 else 0 for x in predictions])

accuracy = accuracy_score(test.get_label(), predict_labels)
print("Accuracy: %.2f%%" % (accuracy * 100.0))