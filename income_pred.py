#!C:/Users/shrey/Anaconda3/python

# coding: utf-8

# Standard Python libraries
import os                                    # For accessing operating system functionalities
import json                                  # For encoding and decoding JSON data
import cgi, cgitb                            # For CGI handling

# Libraries installed via pip
import requests                              # Simple Python library for HTTP
import pandas as pd                          # Library for building dataframes similar to those in R
import seaborn as sns                        # Statistical visualization library based on Matplotlib
import matplotlib
import matplotlib.pyplot as plt              # MATLAB-like plotting, useful for interactive viz

# Utilities from Scikit-Learn, a robust, comprehensive machine learning library in Python.
from sklearn.pipeline import Pipeline
from sklearn.datasets.base import Bunch
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]

# Note: We use regex here for the separator below because the test data
# has a period appended to the end of the class names.
data = pd.read_csv('data/adult.data', sep="\s*,", names=names, engine='python')
#print (data.head())


sns.countplot(y='occupation', hue='income', data=data,)
sns.plt.savefig('occu_count_plot.png')



sns.countplot(y='education', hue='income', data=data,)
sns.plt.savefig('edu_count_plot.png')



g = sns.FacetGrid(data, col='race', size=4, aspect=.5)
g = g.map(sns.boxplot, 'income', 'education-num')
plt.savefig('edunum_race_plot.png')



g = sns.FacetGrid(data, col='sex', size=4, aspect=.5)
g.map(sns.boxplot, 'income', 'education-num')
plt.savefig('edunum_sex_income_plot.png')



g = sns.FacetGrid(data, col='race', size=4, aspect=.5)
g = g.map(sns.boxplot, 'income', 'age')
plt.savefig('age_race_income_plot.png')



g = sns.FacetGrid(data, col='marital-status', size=4, aspect=.5)
g = g.map(sns.boxplot, 'income', 'hours-per-week')
plt.savefig('hours_mari_income_plot.png')


sns.violinplot(x='sex', y='education-num', hue='income', data=data, split=True, scale='count')
plt.savefig('edunum_sex.png')

sns.violinplot(x='sex', y='hours-per-week', hue='income', data=data, split=True, scale='count')
plt.savefig('hours_sex.png')

sns.violinplot(x='sex', y='age', hue='income', data=data, split=True, scale='count')
plt.savefig('age_sex.png')



meta = {
    'target_names': list(data.income.unique()),
    'feature_names': list(data.columns),
    'categorical_features': {
        column: list(data[column].unique())
        for column in data.columns
        if data[column].dtype == 'object'
    },
}

with open('data/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)



def load_data(root='data'):
    # Load the meta data from the file
    with open(os.path.join(root, 'meta.json'), 'r') as f:
        meta = json.load(f)

    names = meta['feature_names']

    # Load the training and test data, skipping the bad row in the test data
    train = pd.read_csv(os.path.join(root, 'adult.data'), names=names)
    test  = pd.read_csv(os.path.join(root, 'adult.test'), names=names, skiprows=1)

    # Remove the target from the categorical features
    meta['categorical_features'].pop('income')

    # Return the bunch with the appropriate data chunked apart
    return Bunch(
        data = train[names[:-1]],
        target = train[names[-1]],
        data_test = test[names[:-1]],
        target_test = test[names[-1]],
        target_names = meta['target_names'],
        feature_names = meta['feature_names'],
        categorical_features = meta['categorical_features'],
    )

dataset = load_data()


gender = LabelEncoder()
gender.fit(dataset.data.sex)


class EncodeCategorical(BaseEstimator, TransformerMixin):
    """
    Encodes a specified list of columns or all columns if None.
    """

    def __init__(self, columns=None):
        self.columns  = columns
        self.encoders = None

    def fit(self, data, target=None):
        """
        Expects a data frame with named columns to encode.
        """
        # Encode all columns if columns is None
        if self.columns is None:
            self.columns = data.columns

        # Fit a label encoder for each column in the data frame
        self.encoders = {
            column: LabelEncoder().fit(data[column])
            for column in self.columns
        }
        return self

    def transform(self, data):
        """
        Uses the encoders to transform a data frame.
        """
        output = data.copy()
        for column, encoder in self.encoders.items():
            output[column] = encoder.transform(data[column])

        return output

encoder = EncodeCategorical(dataset.categorical_features.keys())
data = encoder.fit_transform(dataset.data)



class ImputeCategorical(BaseEstimator, TransformerMixin):
    """
    Encodes a specified list of columns or all columns if None.
    """

    def __init__(self, columns=None):
        self.columns = columns
        self.imputer = None

    def fit(self, data, target=None):
        """
        Expects a data frame with named columns to impute.
        """
        # Encode all columns if columns is None
        if self.columns is None:
            self.columns = data.columns

        # Fit an imputer for each column in the data frame
        self.imputer = Imputer(missing_values=0, strategy='most_frequent')
        self.imputer.fit(data[self.columns])

        return self

    def transform(self, data):
        """
        Uses the encoders to transform a data frame.
        """
        output = data.copy()
        output[self.columns] = self.imputer.transform(output[self.columns])

        return output


imputer = ImputeCategorical(['workclass', 'native-country', 'occupation'])
data = imputer.fit_transform(data)



# We need to encode our target data as well
yencode = LabelEncoder().fit(dataset.target)

# Construct the pipeline
census = Pipeline([
        ('encoder',  EncodeCategorical(dataset.categorical_features.keys())),
        ('imputer', ImputeCategorical(['workclass', 'native-country', 'occupation'])),
        ('classifier', LogisticRegression())
    ])

# Fit the pipeline
census.fit(dataset.data, yencode.transform(dataset.target))


# encode test targets
y_true = yencode.transform([y.rstrip(".") for y in dataset.target_test])

# use the model to get the predicted value
y_pred = census.predict(dataset.data_test)


import numpy as np
from matplotlib import cm

def plot_classification_report(cr, title=None, cmap=cm.YlOrRd):
    title = title or 'Classification report'
    lines = cr.split('\n')
    classes = []
    matrix = []

    for line in lines[2:(len(lines)-3)]:
        s = line.split()
        classes.append(s[0])
        value = [float(x) for x in s[1: len(s) - 1]]
        matrix.append(value)

    fig, ax = plt.subplots(1)

    for column in range(len(matrix)+1):
        for row in range(len(classes)):
            txt = matrix[row][column]
            ax.text(column,row,matrix[row][column],va='center',ha='center')

    fig = plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(len(classes)+1)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.savefig('classif_report.png')
    

cr = classification_report(y_true, y_pred, target_names=dataset.target_names)
#print (cr)
plot_classification_report(cr)

	
# Create instance of FieldStorage 
form = cgi.FieldStorage() 

#Get data from fields
def predict():
	data = {} # Store the input from the user
	data['age'] = ' ' + form.getvalue('age')
	data['workclass']  = ' ' + form.getvalue('workclass')
	data['fnlwgt'] = ' ' + form.getvalue('fnlwgt')
	data['education']  = ' ' + form.getvalue('education')
	data['education-num'] = ' ' + form.getvalue('education-num')
	data['marital-status']  = ' ' + form.getvalue('marital-status')
	data['occupation'] = ' ' + form.getvalue('occupation')
	data['relationship']  = ' ' + form.getvalue('relationship')
	data['race'] = ' ' + form.getvalue('race')
	data['sex']  = ' ' + form.getvalue('sex')
	data['capital-gain'] = ' ' + form.getvalue('capital-gain')
	data['capital-loss']  = ' ' + form.getvalue('capital-loss')
	data['hours-per-week'] = ' ' + form.getvalue('hours-per-week')
	data['native-country']  = ' ' + form.getvalue('native-country')
	# Create prediction and label
	yhat = census.predict(pd.DataFrame([data]))
	return yencode.inverse_transform(yhat)[0]

final_out = predict()

#print to the webpage
print ("Content-type:text/html\r\n\r\n")
print ("<html>")
print ("<head>")
print ("<title>Test</title>")
print ("</head>")
print ("<body>")
print ("<h1>Predicted Income: %s</h1>" %(final_out))
print ("<br/>")
print ("<h3>Classification Report</h3>")
print ("<img src=\"classif_report.png\" style=\"width:850px;height:600px;\">")
print ("<h3>Occupation Count Plot</h3>")
print ("<img src=\"occu_count_plot.png\" style=\"width:850px;height:600px;\">")
print ("<h3>Education Count Plot</h3>")
print ("<img src=\"edu_count_plot.png\" style=\"width:850px;height:600px;\">")
print ("<h3>Number of yrs of Education vs Sex, Income Plot</h3>")
print ("<img src=\"edunum_sex_income_plot.png\" style=\"width:600px;height:600px;\">")
print ("<h3>Number of yrs of Education vs Race, Count Plot</h3>")
print ("<img src=\"edunum_race_plot.png\" style=\"width:1300px;height:600px;\">")
print ("<h3>Age, Race Income Plot</h3>")
print ("<img src=\"age_race_income_plot.png\" style=\"width:1300px;height:600px;\">")
print ("<h3>Hours worked vs Sex Plot</h3>")
print ("<img src=\"hours_sex.png\" style=\"width:2100px;height:600px;\">")
print ("<h3>Hours worked vs Income Plot</h3>")
print ("<img src=\"hours_mari_income_plot.png\" style=\"width:2100px;height:600px;\">")
print ("</body>")
print ("</html>")