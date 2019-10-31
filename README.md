# learn-xgboost
Binary classification example code and data for xgboost.  If you fork this repository and work through all the exercises in this README you can earn the Machine Learning micro-badge.  The example training code here is given in Python and Java, but we will focus on Python for training.  The feature development will be in Java.  So you can open this in Intellij, but you will need to have Python installed.

#### Installation

You need to have [python](https://www.python.org/) which includes package manager pip.  Mac comes with Python2, but we need python 3.7 or newer.  If you use brew to install python3, then python3 is an alias you can use.  We will also need some graphing tools for analysis.  To install python, [xgboost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn), graphing and tracking tools and scikit-learn on Mac:

     brew install python3 graphviz
     pip3 install scikit-learn xgboost mlflow matplotlib
     
On Windows you can [install Python from the Microsoft Store](https://www.microsoft.com/en-us/p/python-37/9nj46sx7x90p?activetab=pivot:overviewtab) and then run the pip3 install command above.  On Linux you will want to install similar package names with your package manager.

#### Training a model with Python
To train a classifier and test that you have installed the main depdendencies you would run:

     python3 py/train-simple.py

#### Labeled data

There are 2660 pairs in csv format in the data directory 1193 neg (start with "0"), and 1467 pos (start with "1").  The fields then alternate after the good (1) or bad (0) label and include person name, spouse name, father name, mother name, child name, person birth detail (year, month, day, country, county, city, location), spouse birth detail (7 fields), child birth detail (7 fields), census detail (year, month, day, country, county, city, location), and death detail (year, month, day, country, county, city, location).  The last field is a reference id.

We will compare each piece of data after the first field (the label), and generate a feature vector for each row.  Those feature vectors are then separated into a training (85%) and a test (15%) set.  Here is a sample row to consider:

0,Joseph Doherty,Joseph Edward John Doherty,Mabel Doherty,Mabel Ruth Weaver,,Joseph Edward Doherty,,Clara Welch,Warren Doherty,Clara Henrietta Doherty,1890,1890,0,7,0,24,1,1,22,22,0,711,0,904814,1890,1898,0,9,0,1,1,1,22,23,0,804,0,1084599,1924,1920,0,5,0,30,1,1,22,22,0,711,0,904814,1940,,0,,0,,1,,22,,901397,,904622,,,1971,,5,,16,,1,,22,,711,,904814,,,,,,,,,,,,,,,,,,,10760990

We need to represent the pair similarities and differences as numbers.  Here are some question to consider:
* How should we generate features from this data?
* What if data is missing?
* What about partial name parts matching?
* Do I need to represent data being close or missing?

Xgboost by default considers missing data to be zero, and you will see that libsvm formatted vectors in sparse form only show the non-zero features.  To start you need to first review the [xgboost tutorial and training data](https://xgboost.readthedocs.io/en/latest/get_started.html) - including the [Text Input Format](https://xgboost.readthedocs.io/en/latest/tutorials/input_format.html) details describing DMatrix input.

An example of libsvm format can be found in learn-xgboost/data directory.  There is also a features.txt file in the data directory that describes each pairing in the .csv data file.  Please review these files in the data directory.

## Micro-badge Introduction
What is xgboost (Extreme Gradient Boosting)?
Xgboost is a popular machine learning algorithm.  It creates boosted decision trees, and has a number of performance and optimization techniques built in.  There is good support and documentation at https://xgboost.ai/ and language support for Python, Java, R, and others.  There are many different algorithms and approaches to machine learning, depending on the problem you want to solve - but we will focus on xgboost in this badge.  To read about other algorithms and solutions scikit-learn is a good resource.  Within scikit you can easily experiment with different machine learning methods - see the API docs for examples.
Here are some ingredients for success:
* Domain knowledge: expertise to distinguish good and bad matches
* Labeled data: good and bad examples that represent well your domain space
* Features: numerical value derived from data comparisons

*Data science superpowers:* Combination of domain knowledge, data collection and aggregation skills, and machine learning tuning and analysis.

We will be focussed on genealogy as the domain in this badge, and you are going to start with a set of labeled data.  You will be building a binary classifier in this badge, developing features, adjusting parameters and interpreting analytic feedback.  This is an example of supervised learning, where you have labeled data to start with, but often more than half of the machine learning problem is gathering or finding good, representative labeled data.
Different algorithms have different strengths and weaknesses, and often you can combine more than one classifier together in an ensemble to gain additional strength.  The gradient boosting method xgboost uses is like an ensemble of classifiers, but in the end the decision tree can be represented as a single tree.  One strength here is you can inspect the decision tree, and see feature importance and the decision flow.  This is useful when trying to understand what the model you build is learning.  Another cool feature in xgboost is the model stores each training iteration, and you can call up any one later with the stored model
We will spend quite a bit of time working with analytic tools and parameters to optimize the model we build.  Most of the parameters available are to help xgboost avoid overfitting.  The gradient boosting degradation is a main theme, where each subsequent training iteration can boost the previous one, but has to have a degradation function for subsequent iterations.  We will also experiment a little of the balance of complexity in the model compared to effectiveness as we adjust some parameters.
There is a workflow that becomes evident in building a good model.  And there is a need to carefully track experiments and measurements along the way.  You will see this workflow in the exercises:

* Build features to expose your problem to the machine
* Create feature vectors
* Train a model
* Analyze the accuracy and other metrics
* Change or add features
* Tune parameters
* Add or adjust training data

It's somewhat magical when machine learning is done right - your smart speaker understanding a difficult sentence, you phone unlocking in poor lighting and at a challenging angle, targeted products or services that appear in your social media stream, etc.  Understanding what is behind the magic takes some practice, but can be fun and eye opening.

### Getting going with feature vectors
To start with you can review and then run CreateFeatureVectors1.java to generate the initial feature vectors.  This implmentation is simply comparing the values as strings and giving a 1 for exact matches.  Notice it divides the input data into 2 
Activity: Compare the output and number of lines in the data directory for the following files:
javaVector
javaVector_eval.libsvm
javaVector_train.libsvm

Question: What percentage of input ends up in the vector train and eval files?

### Training a model
We want to train our first model and assess the accuracy of the model against the test set.  To do this you can run the py/train-simple.py script.  First review the script, and note you need to the the necessary dependencies installed (see above).  Next to train the model run:

     python3 py/train-simple.py
     
Notice we are loading the feature_names, and then the train and eval vector files.  Now would be a good time to review the [xgboost train API documentation](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training).  Notice that the classifier output from train is used to predict against the test set.  The predictions are then labeled depending on their output probability, and the false-positives and false-negatives are computed by comparing those labels with the input labels.

Question: What is the accuracy of your model and how is accuracy computed?

### Building and refining features
We really need to understand the data we are trying to represent to the machine to build the best model.  In looking at this data sample how can we better compare the target and sample?

Joseph Doherty vs Joseph Edward John Doherty (target and candidate)
Mabel Doherty vs Mabel Ruth Weaver (spouses)
Birth detail: 1890 vs 1890 (year), 0 vs 7 (month), 0 vs 24 (day)
1 vs 1 (country), 22 vs 22 (state), 0 vs 711 (county), 0 vs 904814 (city code)
Spouse birth detail: ...
Child birth detail: ...
Census detail: ...

Notice that year/month/day are broken into separate fields.  Why would this be better than just representing a single date?  Notice that the country/state/county/city data is given in codes.  Why is that necessary if we want to compare them?  Next review and run the CreateFeatureVectors2.java file.

Question: How did the output vectors change?

Train the model with the new vectors.

Question: What was the accuracy with the improved name comparison?

Consider these questions:
* Remember we just compared all fields exactly.  What could we do better when representing the data to the classifier?
* Are there some fields that you know should be compared differently?
* How much do improved features help the model?  Will we need to do additional parameter optimization if the features improve?
* What will happen to feature importance if we improve the name feature?

So far we have made a few observations.  Simply comparing name strings doesn't best represent the data to the machine.  Data needs to be normalized and bucketized as we build features.  We have only represented optimistic features to the classifier - patterns to learn that mean you are the same.  Next we will uncomment the differentNames clause in the code.  

Question: What will happen with the differentNames clause uncommented?

Create the new vectors and train the model.

Question: Was the accuracy improved?  How much?  How is this implementation flawed?

Next consider the dates in the input data.  Close inspection will show that often birth dates are estimates, based on age declared at the time the record is made.  So we need to come up with a way to represent year alignment that is close - and let the machine learn which cases it should pay attention to that.  We have already called out the date fields in CreateFeatureVectors2.java (DATE_FIELDS = {10, 24, 38, 66}).  Please review the code commented out that does more than a string compare for the date fields.

Remember that 0 is the default value, and means no data.  So how will we represent year alignment, and the differences?

Question: What value will the date fields contain if the year differs by 1?  What about if they differ by 2, or 5?

Please run the CreateFeatureVectors2.java file after uncommenting the code that compares the years into buckets and train a new model.

Question: What is the accuracy now?  How could you improve this bucketing?

### Analyzing the model
We have been using the accuracy at 50% probability to measure our progress by running the train-simple.py script, but there are lots of other ways to analyze our model.  Also, accuracy might be better if we choose to look at a different probability threshold.  Please take some time and review the train.py script.  Can you see it contains the same train and predict from the train-simple.py script, but there are additional modules loaded.  There is logging and artifact collection using mlflow.  There are additional analysis tools setup - and we have the results at each training iteration.

To start with you can see we will graph the accuracy at each training iteration.  Please train the same vectors you trained above, but this time use the py/train.py script instead of the py/train-simple.py script.

The train.py script will present a number of graphs: Error rates for each iteration, Logloss for each training iteration, Precision/Recall curve for your model, feature importance for your model, and a the decision tree from your model.  (Note: You can close each graph as it pops up, and the next graph will appear.)

We haven't discussed overfitting yet, but your simple training output showed that the error rate of your test set actually started getting worse during training.  The Error rates graph gives you some idea of where your test set error rates stopped getting better.

If you are not familiar with logloss, or the early stopping rounds feature of xgboost please read this article about [Avoiding Overfitting by Early Stopping Rounds in Xgboost](https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/).  The [learning task parameters](https://xgboost.readthedocs.io/en/latest//parameter.html#learning-task-parameters) section describes a number of metrics available during training.  We are going to use logloss as a way pinpoint where overfitting is predicted by the test set.

Xgboost has an [early stopping rounds](https://xgboost.readthedocs.io/en/latest//python/python_api.html#module-xgboost.training) parameter we can use during training.  We tell xgboost to watch back some number of iterations, and let us know when the last set in our watch list stops improving.  Find the commented out 'early_stopping_rounds' code, and exchange it for the existing train line above it.  Try your training again.

Question: What iteration did your model stop on?  Was your accuracy improved?

### Adjusting parameters
We 

### Building your data-science superpowers
