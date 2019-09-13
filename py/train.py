#!/usr/bin/python

import sys
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import mlflow

mlflow.set_experiment("matching")
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

with mlflow.start_run():
  # load the feature names
  with open("data/features.txt", "r") as f:
    feature_names = f.read().split("\n")[:-1]

  train = xgb.DMatrix('data/javaVector_train.libsvm', feature_names=feature_names)
  test = xgb.DMatrix('data/javaVector_eval.libsvm', feature_names=feature_names)

  eval_metrics = ['error', 'logloss']
  num_boost_rounds = 1000
  mlflow.log_param("boost_rounds", num_boost_rounds)

  params = {'n_jobs': 4,
            'eta': 0.1,
            'max_depth': 4,
            'gamma': 0,
            'subsample': 1,
            'colsample_bytree': 1,
            'eval_metric': eval_metrics,
            'objective': 'binary:logistic'}
  mlflow.log_params(params)

  results = {}
  watchlist = [(train, 'train'), (test, 'test')]
  clf = xgb.train(params, train, num_boost_rounds, watchlist, evals_result = results, verbose_eval=100)
  # clf = xgb.train(params, train, num_boost_rounds, watchlist, early_stopping_rounds=20, evals_result = results, verbose_eval=100)

  #####################################
  # Evaluate predictions
  # ...................................
  print("Calculating predictions")
  predictions = clf.predict(test)
  predict_labels = np.array([1 if x >= 0.6 else 0 for x in predictions])

  accuracy = accuracy_score(test.get_label(), predict_labels)
  print("Accuracy: %.2f%%" % (accuracy * 100.0))

  # average_precision = average_precision_score(test.get_label(), predictions)
  # print("Average precision score: %.2f%%" % (average_precision * 100.0))

  mlflow.log_metric("accuracy", accuracy)
  mlflow.log_metrics(clf.get_fscore())

    # Save the last error rate seen for each eval set
  for eval_set in ['train', 'test']:
    mlflow.set_tag(eval_set + " error", results[eval_set]['error'][-1])

  # plot classification error and log loss
  epochs = len(results['train']['error'])
  x_axis = range(0, epochs)
  for metric_name in eval_metrics:
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['train'][metric_name], label='Train')
    ax.plot(x_axis, results['test'][metric_name], label='Test')
    ax.legend()
    plt.ylabel('Classification ' + metric_name)
    plt.title('XGBoost Classification ' + metric_name)
    plt.savefig("artifacts/" + metric_name + ".png")
    plt.show()

  ##################################
  # Plot decision tree, feature importance stack and precision-recall curve
  # ................................
  precision, recall, _ = precision_recall_curve(test.get_label(), predictions)
  plt.step(recall, precision, color='b', alpha=0.2, where='post')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall curve')
  plt.savefig("artifacts/PR_curve.png")
  plt.show()

  plt.rcParams["figure.figsize"] = (14,7)
  xgb.plot_importance(clf, grid=False)
  plt.savefig("artifacts/importance.png")
  plt.show()

  plt.rcParams["figure.figsize"] = (14,3)
  xgb.plot_tree(clf)
  plt.title('Decision Tree')
  plt.savefig("artifacts/tree.png")
  plt.show()

  # persist the model and save artficats to mlflow
  clf.save_model('artifacts/xgb.model')
  mlflow.log_artifacts("artifacts")
