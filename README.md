# learn-xgboost
Binary classification example code and data for xgboost.  If you fork this repository and work through all the exercises in this README you can earn the Machine Learning micro-badge (exercises and questions below).  The example training code here is given in Python and Java, but we will focus on Python for training.  The feature development will be in Java.  So you can open this project in Intellij, but you will need to have Python installed.  You do not need to know much Python for the micro-badge, as mostly you will just be adjusting parameters in Python.

### Installation

You need to have [python3](https://www.python.org/) installed which includes package manager pip.  Mac comes with Python2, but we need python 3.7 or newer.  If you use brew to install python3, then python3 is an alias you can use.  We will also need some graphing tools for analysis.  To install python, [xgboost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn), graphing and tracking tools on Mac:

```shell
brew install python3 graphviz;
pip3 install scikit-learn xgboost mlflow matplotlib graphviz;
```
     
On Windows you can [install Ubuntu from the Microsoft Store](https://www.microsoft.com/en-us/p/ubuntu/9nblggh4msv6) and then launch an Ubuntu terminal:

```shell
apt-get update;
apt-get install python3-pip graphviz;
pip3 install scikit-learn xgboost mlflow matplotlib graphviz;
```

On other flavors of Linux you will want to install similar package names with your package manager.

### Building the project

This repository is set up as a [Maven](https://maven.apache.org/) project. All Java files can be run through IntelliJ or a similar IDE by creating a new project using the included pom.xml file. For users without an IDE that supports Maven, the Maven command line tool can be used to build a jar file through which the Java code can be run. The following Maven command will generate the jar file in the `target` directory:

```shell
mvn clean install
```

### Training a model with Python
To train a classifier and test that you have installed the main dependencies you can run:

```shell
python3 py/train-simple.py
```

You should see output similar to
```
[0]     train-error:0.45088     train-logloss:0.66646   test-error:0.43500      test-logloss:0.66618
[100]   train-error:0.19956     train-logloss:0.42016   test-error:0.21250      test-logloss:0.48054
...
[999]   train-error:0.17345     train-logloss:0.34814   test-error:0.24000      test-logloss:0.51468
Calculating predictions
33 false positives, 63 false negatives
Accuracy: 76.00%
```

You might get the following error:
```
XGBoost Library (libxgboost.dylib) could not be loaded.
Likely causes:
  * OpenMP runtime is not installed
    - vcomp140.dll or libgomp-1.dll for Windows
    - libomp.dylib for Mac OSX
    - libgomp.so for Linux and other UNIX-like OSes
    Mac OSX users: Run `brew install libomp` to install OpenMP runtime.
```

Running this should fix it (on Mac):

```shell
brew install libomp
```

### Labeled data

There are 2660 pairs in csv format in the data directory: 1193 negative "not a match" examples (start with "0"), and 1467 positive "matching" examples (start with "1").  Alternating data fields are given after the good (1) or bad (0) label and include person name, spouse name, father name, mother name, child name, person birth detail (year, month, day, country, state, county, city), spouse birth detail (7 fields), childbirth detail (7 fields), census detail (year, month, day, country, state, county, city), and death detail (year, month, day, country, state, county, city).  The last field is a reference id.

We will compare each piece of data after the first field (the label), and generate a feature vector for each row.  Those feature vectors are then separated into a training (85%) and a test (15%) set.  Here is a sample row to consider:

     0,Joseph Doherty,Joseph Edward John Doherty,Mabel Doherty,Mabel Ruth Weaver,,Joseph Edward Doherty,
     ,Clara Welch,Warren Doherty,Clara Henrietta Doherty,1890,1890,0,7,0,24,1,1,22,22,0,711,0,904814,
     1890,1898,0,9,0,1,1,1,22,23,0,804,0,1084599,1924,1920,0,5,0,30,1,1,22,22,0,711,0,904814,
     1940,,0,,0,,1,,22,,901397,,904622,,,1971,,5,,16,,1,,22,,711,,904814,,,,,,,,,,,,,,,,,,,10760990

We need to represent the pair similarities and differences as numbers.  Here are some question to consider:
* How should we generate features from this data?
* What if data is missing?
* What about partial name parts matching?
* Do I need to represent close or missing data?

Xgboost by default considers missing data to be zero, and you will see that libsvm formatted vectors in sparse form only show the non-zero features.  To start you need to first review the [xgboost tutorial and training data](https://xgboost.readthedocs.io/en/latest/get_started.html) - including the [Text Input Format](https://xgboost.readthedocs.io/en/latest/tutorials/input_format.html) details describing DMatrix input.

An example of libsvm format can be found in learn-xgboost/data directory.  There is also a features.txt file in the data directory that describes each pairing in the .csv data file.  Please review these files in the data directory.

## Micro-badge
To get started you need to fork this repository.  To complete this micro-badge please edit this file and answer the 15 questions in sections A-G.  Then when you are done check-in your work to be reviewed.  The actions are called out like this:

     Fork this repo.

### What is xgboost (Extreme Gradient Boosting)
Xgboost is a popular machine learning algorithm.  It creates boosted decision trees, and has a number of performance and optimization techniques built in.  There is good support and documentation at https://xgboost.ai/ and language support for [Python, Java, R, and others](https://xgboost.readthedocs.io/en/latest//index.html).  There are many different algorithms and approaches to machine learning, depending on the problem you want to solve - but we will focus on xgboost in this badge.  To read about other algorithms and solutions [scikit-learn](https://scikit-learn.org/stable/index.html) is a good resource.  Within scikit you can easily experiment with different machine learning methods.  We will use a few analysis tools from scikit, but see the [scikit API docs](https://scikit-learn.org/stable/modules/classes.html) for examples and documentation.

Here are some ingredients for success:
* Domain knowledge: expertise to distinguish good and bad matches
* Labeled data: good and bad examples that represent well your domain space
* Features: numerical value derived from data comparisons

**Data science superpowers:** Combination of domain knowledge, data collection and aggregation skills, and machine learning tuning and analysis.

We will be focused on genealogy as the domain in this badge, and you are going to start with a set of labeled data.  You will be **building a binary classifier, developing features, adjusting parameters and interpreting analytic feedback**.  This is an example of [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning), where you must start with labeled data.  Often most of the work is gathering good, representative data for your problem.  (We provide the labeled data for the micro-badge.)

Different algorithms have different strengths and weaknesses, and often you can combine more than one classifier together in an ensemble to gain additional strength.  The [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) in xgboost ends up as an ensemble of classifiers.  One benefit of starting with decision trees is there are great tools to visualize things like feature importance or decision flow (for example see [this article on interpreting feature importance](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27)).  This is useful when trying to understand what the model you build is learning.

We will spend time working with analytic tools and parameters to optimize the model we build.  Most of the parameters available are to help xgboost avoid overfitting.  In gradient boosting each subsequent training iteration adds to the previous one, so there is a degradation function for subsequent iterations.  We will be looking at a few parameters like the maximum depth of the tree, and looking at overfitting.  But this is just a quick introduction to the main concepts to get you going with xgboost.

There is a workflow that becomes evident in building a good model.  And there is a need to carefully track experiments and measurements along the way.  You will see this workflow in the exercises:

* Build features to expose your problem to the machine
* Create feature vectors
* Train a model
* Analyze the accuracy and other metrics
* Change or add features
* Tune parameters
* Add or adjust training data

It's somewhat magical when machine learning is done right - your smart speaker understands you, your phone unlocks in poor lighting, web and social media ads know your preferences, etc.  Understanding what is behind the magic takes some practice, but can be fun and eye-opening.


### A. Getting going with feature vectors
To start with please review CreateFeatureVectors1.java.  This implementation is simply comparing the values as strings and giving a 1 for exact matches. Notice it divides the input data into 3 output files.

     Run the CreateFeatureVectors1.java file to generate output in the data directory.
     Compare the output and number of lines in the data directory for the following files:
     javaVector
     javaVector_eval.libsvm
     javaVector_train.libsvm

*Question 1:* What percentage of input lines end up in the vector train and eval files?


### B. Training a model
We want to train our first model and assess the accuracy of the model against the test set.  To do this you can run the py/train-simple.py script. First review the script, and note you need the necessary dependencies installed (see above).  Next, to train the model, run:

```shell
python3 py/train-simple.py
```
     
Notice we are loading the feature_names, and then the train and eval vector files.  Now would be a good time to review the [xgboost train API documentation](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training).  Notice that the classifier output from train is used to predict against the test set.  The predictions are then labeled depending on their output probability, and the false-positives and false-negatives are computed by comparing those labels with the input labels.

*Question 2:* What is the accuracy of your model and how is accuracy computed?


### C. Building and refining features
We really need to understand the data we are trying to represent to the machine to build the best model.  In looking at this data sample how can we better compare the target and sample?

     Joseph Doherty vs Joseph Edward John Doherty (target and candidate)
     Mabel Doherty vs Mabel Ruth Weaver (spouses)
     Birth detail: 1890 vs 1890 (year), 0 vs 7 (month), 0 vs 24 (day)
     1 vs 1 (country), 22 vs 22 (state), 0 vs 711 (county), 0 vs 904814 (city code)
     Spouse birth detail: ...
     Child birth detail: ...
     Census detail: ...

Notice that year/month/day are broken into separate fields.  Why would this be better than just representing a single date?  Notice that the country/state/county/city data is given in codes.  Why is that necessary if we want to compare them?

What if instead of comparing whole name strings, we looked for the number of name parts that align?  Review how the CreateFeatureVectors2.java file will count up the number of name parts that are the same.

     Run the CreateFeatureVectors2.java file.

*Question 3:* How did the output vectors change?

     Train the model with the new vectors.

*Question 4:* What was the accuracy with the improved name comparison?

Consider these questions:
* Remember we just compared all fields exactly. What could we do better when representing the data to the classifier?
* What are some fields that you know should be compared differently?
* How much do improved features help the model?  Will we need to do additional parameter optimization if the features improve?
* What will happen to feature importance if we improve the name feature?

So far we have made a few observations. Simply comparing name strings doesn't best represent the data to the machine.  Data needs to be normalized and bucketized as we build features. We have only represented optimistic features to the classifier - patterns that register when things are the same. In xgboost the default value 0 is used to represent no data, and if either target or candidate doesn't have data to compare, it's like the data isn't there. 

     Uncomment the differentNames else clause in CreateFeatureVectors2.java

*Question 5:* What will happen with the differentNames clause uncommented?

     Create the new vectors and train the model.

*Question 6:* Was the accuracy improved?  How much?  How is this implementation flawed?

Next consider the dates in the input data.  Close inspection will show that often birth dates are estimates, based on age declared at the time the record is made.  So we need to come up with a way to represent year alignment that is close - and let the machine learn which cases it should pay attention to that.  We have already called out the date fields in CreateFeatureVectors2.java (DATE_FIELDS = {10, 24, 38, 66}).

     Review the code in CreateFeatureVectors2.java (commented out) that does more than a string compare for the date fields.

Remember that 0 is the default value, and means no data.  So how will we represent year alignment, and the differences?

*Question 7:* What value will the date fields contain if the year differs by 1?  What about if they differ by 2, or 5?

     Run CreateFeatureVectors2.java after uncommenting the code that compares the years into buckets and train a new model.

*Question 8:* What is the accuracy now?  How could you improve this bucketing?


### D. Analyzing the model
We have been using the accuracy at 50% probability to measure our progress by running the train-simple.py script, but there are lots of other ways to analyze our model.  Also, accuracy might be better if we choose to look at a different probability threshold.  Please take some time and review the train.py script.  You will see it contains the same train and predict from the train-simple.py script, but there are additional modules loaded.  There is logging and artifact collection using mlflow.  There are additional analysis tools setup - and we have the results at each training iteration.

To start with you can see we will graph the accuracy at each training iteration.

     Train again by running the py/train.py script instead of py/train-simple.py

The train.py script will present a number of graphs: Error rates for each iteration, Logloss for each training iteration, Precision/Recall curve for your model, feature importance for your model, and the decision tree from your model's first iteration.  (Note: You can close each graph as it pops up, and the next graph will appear.  If you are using Windows and WSL the graphs will be created in your artifacts directory.)  Please review each graph and spend some time on the feature importance, and tree plot graphs.

*Question 9:* What are the three most important features of your model?

*Question 10:* Where in the first decision tree plot do you expect your most important feature to be?  Where is it actually, and why?

Note: You can modify the 'num_trees' argument in xgb.plot_tree to see the decision tree plot for other iterations.  This gives you some idea of what is happening with each new iteration.  The classifier runs over all the decision trees by default - but you can run it only up to any number of trees in your saved model with additional parameters.

We haven't discussed overfitting yet, but your simple training output showed that the error rate of your test set actually started getting worse during training.  The Error rates graph gives you some idea of where your test set error rates stopped getting better.

If you are not familiar with log loss, or the xgboost 'early stopping rounds' feature you can read this article about [Avoiding Overfitting by Early Stopping Rounds in Xgboost](https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/).  Log loss is one of the eval metrics we have specified in the train.py script.  It is better than just the error rate for evaluating overfitting because it adds a logarithmic penalty factoring in the probability difference for each error.  So really high probability bad examples are penalized more.  There are many other metrics we can track during training.  The [learning task parameters](https://xgboost.readthedocs.io/en/latest//parameter.html#learning-task-parameters) section describes logloss and other metrics.  We are going to use logloss as a way pinpoint which iteration starts overfitting - or in other words where our test set stops improving, even though our training set is still improving.

Xgboost has an [early stopping rounds](https://xgboost.readthedocs.io/en/latest//python/python_api.html#module-xgboost.training) parameter we can use during training.  We tell xgboost to watch back some number of iterations, and let us know when the last set in our watch list stops improving.  

     Find the 'early_stopping_rounds' line in train.py, and exchange that line for the one above it.  Try your training again.

*Question 11:* What iteration did your model stop on?  Was your accuracy improved?

The Precision/Recall graph shows the tension between getting all the answers right, and getting answers for all the questions.  You can find an optimal point on the curve where you maximize precision or recall, or maybe you want to balance them.  Then you can find on that curve what the precision threshold is at that point.  If you are interested in pursuing threshold analysis you can look into the data behind the P/R curve.

### E. Adjusting parameters
The number of features you have, the way you represent your data to the machine, the amount of training data, the percentage and representativeness of your test set, and the model parameters all influence your ideal outcome.  The complexity of your machine learning model has an optimum for your problem, and the way you have represented it.  So as the data, or features, or parameters change there are different potential optimization points.  This is important to realize as we try to introduce and adjust a few parameters.  It's possible that just a few adjustments to parameters will best leverage the technology you have chosen.

Most of the parameters for xgboost are set to avoid overfitting, and defaults are conservative.  But xgboost is pretty forgiving, and we won't be able to experiment with all the parameters in this micro-badge.  Here is a short explanation of a few parameters:

* `max_depth`: maximum depth of a tree - increasing may lead to complex tree and overfitting
* `subsample`: subsample ratio of training instances, for example .5 would use only half of the training data before each epoch
* `colsample_bytree`: subsample ratio of columns for each epoch
* `eta`: the learning rate, or the step size shrinkage used to prevent overfitting - each epoch it shrinks the feature weights to make it more conservative
* `gamma`: minimum loss reduction to make further partition on lead node - larger gamma is more conservative

We have called out some basic parameters in the params dictionary in the train.py script.  We have been using:

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

A few pointers on the different xgboost [general parameters](https://xgboost.readthedocs.io/en/latest//parameter.html#general-parameters) and [booster parameters](https://xgboost.readthedocs.io/en/latest//parameter.html#parameters-for-tree-booster): 
* Our sample data is small enough that we don't notice, but for larger datasets xgboost scales very well.  The `n_jobs` can leverage the cores in your computer to train in parallel.  Across the different languages this option changes a bit; for example notice in the Train.java file there is no n_jobs parameter.
* `eta` is also sometimes called the learning rate, and is important to prevent overfitting
* We are using defaults for `subsample` and `colsample_bytree` - which means we use all the training data at each iteration.  Often, you want to generalize your problem across your training data - so these settings see if the system can generalize solutions when some part of the training data is suppressed each iteration.

We are going to experiment with the max_depth option.  This controls the depth of the resulting decision tree, and probably has some affinity to the number of features you are working with.  

     Change the max_depth from 4 to 6.  Retrain and analyze your model.

*Question 12:* How is your accuracy and tree impacted by this change?

     Try adjusting max_depth some more.

*Question 13:* How is your accuracy and tree impacted by this change?  Did you find an optimal max_depth?


### F. Building your data-science superpowers
Doing all these experiments requires some scientific rigor to understand what is changing, and to track results.  Along the way each training run has been logging the results into an artifact directory, but also the parameters, metrics and artifacts have been logged into [mlflow](https://mlflow.org/) locally.  This is an open-source tool for tracking machine learning and experimenting through the development life-cycle.  We will just be demonstrating a small portion of the functionality - logging training parameters, metrics and artifacts.

Notice the train.py code we log parameters, metrics and artifacts to mlflow.  The result is each run has been accumulating that data to your mlruns directory in this project root.  To start up the UI to browse the results run the following in the root directory:

     mlflow ui

This will start up a local mlflow UI at http://127.0.0.1:5000 that you can open in your browser.  Select the `matching` experiment in the left pane if it isn't already selected.  Type `max_depth` in the box labeled 'Filter Params:' and `accuracy` in the box labeled 'Filter Metrics:'.  Click 'Search'.  You should see each of your training runs, and the parameters and metrics recorded.  Select a few runs and click 'Compare' - notice how the feature importance changed between runs.  Return to the list of runs and click on one - scroll down to the artifacts area and notice you can click on the .png artifacts to display the graphs.

*Question 14:* What might be useful here in comparing training runs?


### G. Choose your own adventure
We only scratched the surface of feature development and fitting those features to the machine.  Xgboost allows you to bundle features so they have to stay together, and provide weights to help emphasize what is important.  There are many other ways to better represent our problem, and you might already have noticed a few.  For one, we are only comparing similar fields to each other.  What if the target has a birth date, but the candidate does not - but the candidate has a child marriage date and the target does not.  Could you add a feature that would expose that information to the machine and see if it helps improve things?

     The final activity is to develop a feature or find another way to improve the accuracy of our example code.

*Question 15:* What feature did you develop or optimize?  How much did it help improve your model?

In summary:
* Getting good labeled data to represent your problem is often the hardest thing
* Feature selection and development is important
* Parameters are important in optimizing the model
* Decision trees are fast and pretty forgiving, but also prone to overfitting
* Analysis tools are key in understanding and building good models

Once you are done, you can push up your fork and let benhafen@ChurchofJesusChrist.org or macdonalddm@familysearch.org know you are done so we can review your work.  Congratulations - hopefully you enjoyed experimenting with this sample code.  We hope you can easily plug your own problem in and start experimenting with this powerful framework.
