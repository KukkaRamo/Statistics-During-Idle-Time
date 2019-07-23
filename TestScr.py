# Wheter to save space and repeat calculating x and y every time you need them
# or do it only once to save time

# All dependent variables are dichotomous here (either 0 or 1)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier

# Data order: First continuous variables, then ordinal, and then categorical

myPath = "C:/Users/Kukka/Documents"
fileName = myPath + "/KeksittyTesti.csv"
numOfContinuousInDep = 20
numOfOrdinalInDep = 0
numOfCategoricalInDep = 5
numOfDep = 5

numOfOrderableInDep = numOfContinuousInDep + numOfOrdinalInDep
numOfIndep = numOfOrderableInDep + numOfCategoricalInDep
firstCategorical = numOfOrderableInDep

data=pd.read_csv(fileName)
array = data.values

# numOfKBestK = 3 Limiting SelectKBest features if we want a reduced set from transform
myLogisticSolver = 'liblinear' # my solver and penalties here
numOfRFEBest = 3 #How many features we want to keep in recursive feature elimination
# ridgeAlpha = 1.0 and then 0.1 All dependent variables are binary, so logistic regression is used, no ridge
corSigLimit = 0.9 # How big correlations we want to print, give an absolute value, both signs are processed
numOfRandomForestTrees = 100 # Number of trees in random forest estimation

orderableColList = data.columns.values[0:firstCategorical]
ordinalColList = data.columns.values[numOfContinuousInDep:numOfContinuousInDep + numOfOrdinalInDep]
categoricalColList = data.columns.values[firstCategorical:firstCategorical + numOfCategoricalInDep]
orderableData = array [:,0:firstCategorical]
ordinalData = array [:, numOfContinuousInDep:firstCategorical]
categoricalWithDummies = pd.get_dummies(data.loc[:,categoricalColList])

def LogisticFor (name, numOfRFEBest, cols, xdata, ydata):
    model = LogisticRegression(solver = myLogisticSolver) # Add other parameters
    rfe = RFE(model,numOfRFEBest)
    print ("Logistic regression coef, intercept, and RFE-rankings for " + str (numOfRFEBest) + " features ")
    try:
        print(name)
        print (cols)
        print ("Model coefficients and intercept")
        fit = model.fit(xdata, ydata)
        print (fit.coef_)
        print (fit.intercept_)
        print ("Feature importances in feature elimination")
        fit = rfe.fit(xdata, ydata)
        print (fit.ranking_)
    except ValueError as e:
        print ("Value error in logistic regression for " + name + " " + str(e))

for currentDep in range (numOfIndep, numOfIndep + numOfDep):
    print ("Next round, current dep " + str(currentDep) + " " + data.columns[currentDep])
    y = array [:,currentDep]
    y = y.astype('bool')

    # SelectKBest for categorical, scores and p-values in chi square
    test = SelectKBest(score_func = chi2, k = 'all')
    fit = test.fit(categoricalWithDummies,y)
    print ("SelectKBest of categorical variables ")
    print (categoricalWithDummies.columns)
    print ("Scores of categories ")
    print (fit.scores_)
    print ("p-values for categories ")
    print (fit.pvalues_)

    # SelectKBest for orderable, scores and p-values in f_classif
    test = SelectKBest(score_func = f_classif, k = 'all')
    fit = test.fit(orderableData,y)
    print ("SelectKBest of orderable variables ")
    print (orderableColList)
    print ("Scores of columns ")
    print (fit.scores_)
    print ("p-values for columns ")
    print (fit.pvalues_)
    
    # LogisticRegression for categorical, then ordinal, and then orderable
    LogisticFor("categorical", numOfRFEBest, categoricalWithDummies.columns, categoricalWithDummies, y)
    LogisticFor("ordinal", numOfRFEBest, ordinalColList, ordinalData, y)
    LogisticFor("ordered", numOfRFEBest, orderableColList, orderableData, y) 
    
    # IsotonicRegression for orderable
    for currentInDep in range (0, numOfOrderableInDep):
        model = IsotonicRegression(increasing='auto')
        xn = array [:,currentInDep]
        fit = model.fit(xn,y)
        print ("Isotonic ")
        print (data.columns.values[currentInDep])
        print (fit.score(xn,y))
        print (xn)
        ff = model.fit_transform(xn,y)
        print (ff)

    # For orderable: a correlation between the (product of values of this pair of variables) and the dependent variable
    print ("Studying cross-correlations ")
    for i in range (0, firstCategorical):
        for j in range (i+1, firstCategorical):
            corrx = array [:,i] * array [:,j]
            coefs = np.corrcoef(corrx.astype(float),y.astype(float))
            if abs(coefs [0,1]) > corSigLimit:
                print ('Product significant limit exceeded ')
                print (data.columns.values[i])
                print (data.columns.values[j])
                print (coefs)

    # For orderable: a correlation between the (product of values of the first and inverted second variables) and the dependent variable
    print ("Studying cross-correlations ")
    for i in range (0, firstCategorical):
        for j in range (i+1, firstCategorical):
            corrx = array [:,i] * -array [:,j]
            coefs = np.corrcoef(corrx.astype(float),y.astype(float))
            if abs(coefs [0,1]) > corSigLimit:
                print ('Product significant limit exceeded ')
                print (data.columns.values[i])
                print ("-" + str(data.columns.values[j]))
                print (coefs)

    # DecisionTreeClassifier for categorical
    model = DecisionTreeClassifier()
    fit = model.fit(categoricalWithDummies, y)
    print ("DecisionTreeClassifier ")
    print (categoricalWithDummies.columns)
    print (fit.feature_importances_)
	
    # RandomForestRegressor for categorical
    model = RandomForestRegressor(n_estimators = numOfRandomForestTrees, random_state = 42)
    fit = model.fit (categoricalWithDummies, y)
    print ("RandomForestRegressor ")
    print (categoricalWithDummies.columns)
    print (fit.feature_importances_)
    # print (fit.oob_score_) oob_score is false, just try this 
    # print (fit.warm_start) fit is called only once
    
    # For categorical variables: for all values, is there a binary correlation between this value and the dependent variable
    print ("Categorical variables one by one")
    for i in range (0, len(categoricalWithDummies.columns)):
        coefs = np.corrcoef(categoricalWithDummies.loc[:,categoricalWithDummies.columns.values[i]],y)
        if abs(coefs [0,1]) > corSigLimit:
            print ('Significant limit exceeded ')
            print (categoricalWithDummies.columns.values[i])
            print (coefs)


# Lähteet:
# Sayak, P: Beginner's Guide to Feature Selection in Python 28 Sep 2018, retrieved: Jul 9 2019
# https://www.datacamp.com/community/tutorials/feature-selection-python
