# learn-xgboost
Binary classification example code and data for xgboost.  If you fork this repository and work through all the exercises in this README you can earn the Machine Learning micro-badge.  The example training code here is given in Python and Java, but we will focus on Python for training.  The feature development will be in Java.  So you can open this in Intellij, but you will need to have Python installed.

#### Installation

You need to have [python](https://www.python.org/) which includes package manager pip.  Mac comes with Python2, but we need python 3.7 or newer.  If you use brew to install python3, then python3 is an alias you can use.  We will also need some graphing tools for analysis.  To install python, [xgboost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn), graphing and tracking tools and scikit-learn on Mac:

     brew install python3 graphviz
     pip3 install scikit-learn xgboost mlflow matplotlib
     
On Windows you can [install Python from the Microsoft Store](https://www.microsoft.com/en-us/p/python-37/9nj46sx7x90p?activetab=pivot:overviewtab).  On Linux you will want to install similar package names with your package manager.

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

#### Building features
Please run CreateFeatureVectors1.java to generate the initial features.  Notice that this implmentation is simply comparing the values as strings and giving a 1 for exact matches.

