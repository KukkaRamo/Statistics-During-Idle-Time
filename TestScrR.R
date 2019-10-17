# Wheter to save space and repeat calculating x and y every time you need them
# or do it only once to save time

# All dependent variables are dichotomous here (either 0 or 1)

# Data order: First continuous variables, then ordinal, and then categorical
# All data is included

myPath <- "C:/Users/Kukka/Documents"
fileName <- paste(myPath,"/KeksittyTesti.csv", sep="")
numOfContinuousInDep <- 20
numOfOrdinalInDep <- 0
numOfCategoricalInDep <- 5
numOfDep <- 5

numOfOrderableInDep <- numOfContinuousInDep + numOfOrdinalInDep
numOfIndep <- numOfOrderableInDep + numOfCategoricalInDep

# wrapr::seqi builds an empty sequence if start > end, the default in R is to build a reverse sequence
library (wrapr)

firstOrderable <- 1
lastOrderable <- firstOrderable + numOfOrderableInDep - 1
orderableIndeces <- seqi(firstOrderable, lastOrderable)
firstOrdinal <- numOfContinuousInDep + 1
lastOrdinal <- firstOrdinal + numOfOrdinalInDep - 1
ordinalIndeces <- seqi(firstOrdinal, lastOrdinal)
firstCategorical <- numOfOrderableInDep + 1
lastCategorical <- firstCategorical + numOfCategoricalInDep - 1
categoricalIndeces <- seqi(firstCategorical, lastCategorical)
firstIndependent <- 1
lastIndependent <- firstOrderable + numOfIndep - 1
independentIndeces <- seqi(firstIndependent, lastIndependent)

myOriginalData <- read.csv(file=fileName, header=TRUE, sep=",")
myData <- na.omit(myOriginalData)

categoricalFormulaRS <- paste (colnames(myData)[categoricalIndeces], collapse = "+")
orderableFormulaRS <- paste (colnames(myData)[orderableIndeces], collapse = "+")
ordinalFormulaRS <- paste (colnames(myData)[ordinalIndeces], collapse = "+")
independentFormulaRS <- paste (colnames(myData)[independentIndeces], collapse = "+")

numOfRFEBest = 3 #How many features we want to keep in recursive feature elimination
corSigLimit = 0.9 # How big correlations we want to print, give an absolute value, both signs are processed
importanceRfNumber <- 3
importanceRfRepeats <- 3

orderableColList = colnames(myData)[orderableIndeces]
ordinalColList = colnames(myData)[ordinalIndeces]
categoricalColList = colnames(myData)[categoricalIndeces]
independentColList = colnames(myData) [independentIndeces]
orderableData = myData [,orderableIndeces]
ordinalData = myData [, ordinalIndeces]
categoricalData = myData [, categoricalIndeces]
independentData = myData [, independentIndeces]

#install.packages("fastDummies")
library(fastDummies)
origColumns <- ncol(myData)
dummiesResult <- fastDummies::dummy_columns(myData)
numDummies <- ncol(dummiesResult)
if (numDummies > 0) {
	categoricalWithDummies <- dummiesResult[, (origColumns + 1) : numDummies]
}

function.InitFormula <- function(type, pYname, rs) {
	if (rs=="") {
		return (NULL)
	}
	tryCatch (
		{
			pFormula=NULL
			pFormula <- as.formula(paste(pYname, "~",
				rs,
				sep = ""
			))
			return (pFormula)
		},
		error = function (e)
		{
			print (paste("error in initializing formula",type,yname,e))
			return (NULL)
		}
	)
}

library (ISLR)
function.LogisticFor <- function (name, myFormula, numOfRFEBest, yname, cols, xdata, ydata) {
	print ("In LogisticFor")
	print(name)
    print (cols)
	glm.fit <- glm( formula = myFormula, family = binomial, data = myData, na.action = na.exclude)
    print (paste("Logistic regression coef, intercept, and RFE-rankings for ",numOfRFEBest," features "))
    tryCatch (
		{
			print(name)
			print (cols)
			print ("Model coefficients and intercept glm")
			print (summary(glm.fit))
			print ("glm succeeded")
		},
		# except ValueError as e 
		error = function (e) {
			print (paste("Value error in logistic regression for",name,e))
		}
	)
}

if (numOfDep > 0) {
	for (currentDep in (numOfIndep + 1) : (numOfIndep + numOfDep)) { #Why does this work only with parenthesis
		print (paste("Next round, current dep ",currentDep," ",colnames(myData)[currentDep]))
		y <- myData [,currentDep]
		y <- as.logical(y)
		yname <- colnames(myData)[currentDep]
		
		allCategoricalFormula <- function.InitFormula("categorical", yname, categoricalFormulaRS)
		allIndependentFormula <- function.InitFormula("independent", yname, independentFormulaRS)
		allOrderableFormula <- function.InitFormula("orderable", yname, orderableFormulaRS)
		allOrdinalFormula <- function.InitFormula("ordinal", yname, ordinalFormulaRS)
			
		#install.packages('caret', dependencies = TRUE)

		library(caret)

		# define the control using a random forest selection function
		control <- rfeControl(functions=rfFuncs, method="cv", number=10)
		# run the RFE algorithm
		tryCatch (
			{
				results <- rfe(independentData, factor(y), sizes=c(ncol(independentData)), rfeControl=control)
				# summarize the results
				print(results)
				print("results$fit:")
				print(results$fit)
				print ("Importances")
				print (varImp(results, scale=FALSE))
				print ("rfe succeeded")
			},
			error = function (e) {
				print (paste("Cannot do rfe",yname,e))
			}
		)
		
		# LogisticRegression for categorical, then ordinal, and then orderable
		if ((exists("allCategoricalFormula")) & (!(is.null(allCategoricalFormula)))) {
			function.LogisticFor("categorical", allCategoricalFormula, numOfRFEBest, yname, categoricalColList, categoricalData, y)
		}
		if ((exists("allOrdinalFormula")) & (!(is.null(allOrdinalFormula)))) {
			function.LogisticFor("ordinal", allOrdinalFormula, numOfRFEBest, yname, ordinalColList, ordinalData, y)
		}
		if ((exists("allOrderableFormula")) & (!(is.null(allOrderableFormula)))) {
			function.LogisticFor("ordered", allOrderableFormula, numOfRFEBest, yname, orderableColList, orderableData, y) 
		}
		
		# IsotonicRegression for orderable
		if (exists("orderableIndeces")) {
			for (currentInDep in orderableIndeces) {
				xn = myData [,currentInDep]
				model = isoreg(xn, y)
				print ("Isotonic ")
				print (colnames(myData)[currentInDep])
				print (xn)
				print (model)
			}
		}

		# For orderable: a correlation between the (product of values of this pair of variables) and the dependent variable
		print ("Studying cross-correlations ")
		if (exists("orderableIndeces")) {
			for (i in orderableIndeces) {
				for (j in seqi((i+1), lastOrderable)) {
					result <- cor(myData [,i] * myData [,j] ,y)
					if (!is.na(result)) {
						if (abs(result) > corSigLimit) {
							print ('Product significant limit exceeded ')
							print (colnames(myData)[i])
							print (colnames(myData)[j])
							print (result)
						}
					}
				}
			}
		}

		# For orderable: a correlation between the (product of values of the first and inverted second variables) and the dependent variable
		print ("Studying cross-correlations ")
		if (exists("orderableIndeces")) {
			for (i in orderableIndeces) {
				for (j in seqi ((i+1), lastOrderable)) {
					result <- cor(myData [,i] * (- myData [,j]) ,y)
					if (!is.na(result)) {
						if (abs(result) > corSigLimit) {
							print ('Product significant limit exceeded ')
							print (colnames(myData)[i])
							print (paste("-",colnames(myData)[j]))
							print (result)
						}
					}
				}
			}
		}
		
		if ((exists("allCategoricalFormula")) & (!(is.null(allCategoricalFormula)))) {
			library(party)
			# DecisionTreeClassifier for categorical
			model1 <- ctree(allCategoricalFormula, data = myData)
			print ("DecisionTreeClassifier ")
			print (categoricalColList)
			print (model1)
			
			tryCatch (
				{
					plot(model1)
				},
				error = function (e) {
					print (paste("DecisionTreeClassifier plot",yname,e))
				}
			)

			
			library(randomForest)
			# RandomForestRegressor for categorical
			model <- randomForest(allCategoricalFormula, data = myData)
			print ("RandomForestRegressor ")
			print (categoricalColList)
			print (model)
			print (importance(model))
			print (importance(model, type=1))
			
			tryCatch (
				{
					plot(model)
				},
				error = function (e) {
					print (paste("RandomForestRegressor plot",yname,e))
				}
			)
		}
		
		# For categorical variables: for all values, is there a binary correlation between this value and the dependent variable
		if (exists("categoricalWithDummies")) {
			print ("Categorical variables one by one")
			for (i in 1:(ncol(categoricalWithDummies))) {
				result <- cor(categoricalWithDummies [,i] ,y)
				if (!is.na(result)) {
					if (abs(result) > corSigLimit) {
						print ('Significant limit exceeded ')
						print (colnames(categoricalWithDummies)[i])
						print (result)
					}
				}
			}
		}
	}
}

