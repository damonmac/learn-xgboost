#### Quick start

You need to have [node](http://nodejs.org/) installed to run the features.coffee script.  This will generate features from the raw csv file in this directory.  Once node is installed, you can install dependencies with:

     npm install

To install the coffee command globally run:

     npm -g install coffeescript

To generate features then run:

     coffee features.coffee > vector-exp

If you want to create a features.js file from the features.coffee file run:

     coffee -c features.coffee
