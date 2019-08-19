#!/usr/bin/python

import sys
import mlflow
from mlflow import sklearn
from time import time
sys.path.append("./")
from prepare import preprocess
from sklearn.metrics import accuracy_score, recall_score, precision_score, average_precision_score, precision_recall_curve
from xgboost import XGBClassifier, plot_tree, plot_importance
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = preprocess()

mlflow.set_experiment("matching")
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

with mlflow.start_run():

  params = {'n_estimators': 700, 
            'eta': 0.1, 
            'max_depth': 7, 
            'gamma': 0.95, 
            'subsample': 0.95, 
            'silent': 1,
            'objective': 'binary:logistic'}

  for key in params:
    mlflow.log_param(key, params[key])

  earlyStopping = 10
  mlflow.log_param("early_stopping_rounds", earlyStopping)

  clf = XGBClassifier(**params)

  t0 = time()
  eval_set = [(X_train, y_train),(X_test, y_test)]
  clf.fit(X_train, y_train, early_stopping_rounds=earlyStopping, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
  print("training time:", round(time()-t0, 2), "s")

  # training metrics
  results = clf.evals_result()
  epochs = len(results['validation_0']['error'])
  x_axis = range(0, epochs)

  # plot classification error
  fig, ax = plt.subplots()
  ax.plot(x_axis, results['validation_0']['error'], label='Train')
  ax.plot(x_axis, results['validation_1']['error'], label='Test')
  ax.legend()
  plt.ylabel('Classification Error')
  plt.title('XGBoost Classification Error')
  plt.savefig("Error.png")
  mlflow.log_artifact("Error.png")
  plt.show()

  # plot classification log loss
  fig, ax = plt.subplots()
  ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
  ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
  ax.legend()
  plt.ylabel('Log Loss')
  plt.title('XGBoost Log Loss')
  plt.savefig("Logloss.png")
  mlflow.log_artifact("Logloss.png")
  plt.show()

  # persist the model
  # clf.save_model('xgb.model')
  mlflow.sklearn.log_model(clf, "model")

  t0 = time()
  pred = clf.predict(X_test)
  print("prediction time:", round(time()-t0, 2), "s")

  #####################################
  # Evaluate predictions
  # ...................................
  accuracy = accuracy_score(y_test, pred)
  print("Accuracy: %.2f%%" % (accuracy * 100.0))

  precision = precision_score(y_test, pred)
  print("Precision: %.2f%%" % (precision * 100.0))

  recall = recall_score(y_test, pred)
  print("Recall: %.2f%%" % (recall * 100.0))

  y_score = clf.predict_proba(X_test)[:,1]
  average_precision = average_precision_score(y_test, y_score)
  print("Average precision-recall score: %.2f%%" % (average_precision * 100.0))

  mlflow.log_metric("accuracy", accuracy)
  mlflow.log_metric("precision", precision)
  mlflow.log_metric("recall", recall)
  mlflow.log_metric("average_precision", average_precision)

  importances = clf.get_booster().get_fscore()
  for key in importances:
    mlflow.log_metric(key, importances[key])

  ##################################
  # Plot the tree, importance and Precision-Recall curve
  # ................................
  plot_importance(clf)
  plt.show()
  plot_tree(clf)
  plt.show()
  mlflow.log_artifact("tree.png")

  precision, recall, _ = precision_recall_curve(y_test, y_score)
  plt.step(recall, precision, color='b', alpha=0.2, where='post')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall curve: AP={0:0.4f}'.format(average_precision))
  plt.savefig("PR_curve.png")
  mlflow.log_artifact("PR_curve.png")
  plt.show()