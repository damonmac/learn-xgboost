# learn-xgboost
Binary classification example code and data for xgboost

[Scikit-learn](http://scikit-learn.org/stable/) supports many different classification methods, with good documentation and example code in python.  You might want to look at the [sklearn.svm.SVC docs](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), and the excellent [user guide](https://scikit-learn.org/stable/modules/svm.html#svm-classification).

#### Quick start

You need to have [python](https://www.python.org/) installed (which includes package manager pip).  OSX supports multiple versions of python, but we will need python 3.7 or newer.  If you use brew to install python3, then python3 is an alias you can use.  To install python, [xgboost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) and scikit-learn on OSX try:

     brew install python3
     pip3 install scikit-learn xgboost mlflow

If you want a specific version of scikit the command would be pip3 install scikit-learn==0.20.1 (latest stable).

#### Training and saving a model
To train a classifier you would run:

     python3 train-simple.py

You can run the the python scripts from IntelliJ with the python plug-in.

#### Sample data

There are 2660 pairs in csv format in the data directory 1193 neg (start with "0"), and 1467 pos (start with "1").  The fields then alternate after the label for target, then candidate.  They include person name, spouse, parent1, parent2, child, person birth year, month, day, country, county, city, location, census year, month, day, country, county, city, location, and death year, month, day, country, county, city, location.  The last field is a reference id.
