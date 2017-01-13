# Fraud Detection Case Study

## Premise
You are a contract data scientist/consultant hired by a new e-commerce site to try to weed out fraudsters.  The company unfortunately does not have much data science expertise... so you must properly scope and present your solution to the manager before you embark on your analysis.  Also, you will need to build a sustainable software project that you can hand off to the companies engineers by deploying your model in the cloud.


## Day 1:

### Step 1: EDA

### Step 2: Building the Model

1. Create a file called `model.py` which builds the model based on the training data.

    * Feel free to use any library to get the job done.

2. In your pull request, describe your project findings including:
    * An overview of a chosen “optimal” modeling technique, with:
        * process flow
        * preprocessing
        * accuracy metrics selected
        * validation and testing methodology
        * parameter tuning involved in generating the model
        * further steps you might have taken if you were to continue the project.

#### [Deliverable]: Pickled model

1. Use `cPickle` to serialize your trained model and store it in a file. This is going to allow you to use the model without retraining it for every prediction, which would be ridiculous.


### Step 3: Prediction script

1. Write a script `predict.py` that reads in a single example from `test_script_examples`, vectorizes it, unpickles the model, predicts the label, and outputs the label probability (print to standard out is fine).

    This script will serve as a sort of conceptual and code bridge to the web app you're about to build.

    Each time you run the script, it will predict on one example, just like a web app request. You may be thinking that unpickling the model every time is quite inefficient and you'd be right; we'll remove that inefficiency in the web app.


### Step 4: Database

We want to store each prediction the model makes on new examples, which means we'll need a database.

1. Set up a Postgres or MongoDB database that will store each example that the script runs on. You should create a database schema that reflects the form of the raw example data and add a column for the predicted probability of fraud.

2. Write a function in your script that takes the example data and the prediction as arguments and inserts the data into the database.

    Now, each time you run your script, one row should be added to the `predictions` table with a predicted probability of fraud.


## Day 2:

### Step 5: Web App

#### [Deliverable]: Fraud scoring service

1. Set up a route `POST /score` and have it execute the logic in your prediction script. You should import the script as a module and call functions defined therein.

    There are two things we'll do to make this all more efficient:

    1. We only want to unpickle the model once
    2. We only want to connect to the database once.

    Do both in a `if __name__ == '__main__':` block before you call `app.run()` and you can refer to these top-level global variables from within the function. This may require some re-architecting of your prediction module.

    The individual example will no longer be coming from a local file, but instead will come in the body of the POST request as JSON. You can use `request.data` or `request.json` to access that data. You'll still need to vectorize it, predict, and store the example and prediction in the database.

    You can test out this route by, in a separate script, sending a POST request to /score with a single example in JSON form using the `requests` Python package.


### Step 6: Get "live" data

We've set up a service for you that will ping your server with "live" data so that you can see that it's really working.

To use this service, you will need to make a POST request to `ourcomputersip/register` with your IP and port. We'll announce what the ip address of the service machine is in class.


1. Write a register function which makes the necessary post request. This function should be called once each time your Flask app is run, in the main block.

**Make sure your app is adding the examples to the database with predicted fraud probabilities.**

### Step 7: Dashboard

#### [Deliverable]: Web front end to present results

We want to present potentially fraudulent transactions with their probability scores from our model. The transactions should be segmented into 3 groups: low risk, medium risk, or high risk (based on the probabilities).

* Add route in Flask app for dashboard
* Read data from postgres
* Return HTML with the data
    * To generate the HTML from the json data from the database, either just use simple string concatenation or Jinja2 templates.


### Step 8: Deploy!

* Set up AWS instance
* Set up environment on your EC2 instance
* Push your code to github
* SSH into the instance and clone your repo
* Run Flask app on instance (make sure you update the register code with your updated ip and port)
* Make it work (debug, debug, debug)
* Profits!
