# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 11:35:34 2019

@author: Megha
"""
#Significance Test

# In this test, we are perfrom two types of test - ttest,anova
# ttest  - is used to compare assumed population mean with given sample
# mean or compare two sample mean values.
# anova - is used to compare means of more than two sample.

# ttest is of three types.
# type-1 one sample test - ttest_1samp()
# it is used to compare an assumed mean with sample mean value
# H0: assumed mean = value
# H1: assumed mean not = value

#type-2 two sample test - unpaired ttest - ttest_ind()
# it is used to compare two sample mean value
# H0: sample1 mean = sample2 mean
# H1: sample1 mean not = sample2 mean

#type-3 two sample test - paired ttest - ttest_rel()
# it is used to compare mean of a sample before an event with mean of same sample
# after event.
# H0: before_event_mean - after_event_mean = 0
# H1: before_event_mean - after_event_mean not = 0

# If pvalue is more than or equal to 0.05, we accept H0 else reject H0

# extract data brain_size
# find rows and column count  40,8
import pandas as pd
df=pd.read_csv("C:/Users/Alok/Desktop/Data Science Workshop/Alliance Data/Day 2/Data Files/brain_size.csv",sep=";")
df.shape
df.columns
from scipy import stats

# H0: mean=120
# H1: mean not = 120

stats.ttest_1samp(df['VIQ'],120)
#stats.ttest_1samp(df['VIQ'],108)

# conclusion: statistic=-2.0487224279216516, pvalue=0.047261241275810835
# As pvalue is less than 0.05, so reject H0. Mean not equal to 120.

# statistics ( T-value) = (sample_mean - assumed_mean)/std_error
# std_error = sample_std/sqrt(sample_size)

# sample_mean - assumed_mean < 0 i.e assumed mean is more than sample mean

# two sample test - 
# unpaired ttest
# compare mean VIQ of female with mean VIQ of male
# H0: mean VIQ of female = mean VIQ of male
# H1: mean VIQ of female not = mean VIQ of male

female_VIQ = df[df['Gender']=="Female"]['VIQ']
male_VIQ = df[df['Gender']=="Male"]['VIQ']

stats.ttest_ind(female_VIQ,male_VIQ)
# conclusion: statistic=-0.7726161723275011, pvalue=0.44452876778583217
# As pvalue is more than 0.05, so we accept H0, female and male have equal
# average VIQ.

# paired ttest
# if FSIQ is IQ score before training and PIQ is IQ score after training.
# i am interested to know the training program has improve IQ score of 
#individual or not

# H0: FSIQ-PIQ=0, H1: FSIQ-PIQ not =0
stats.ttest_rel(df['FSIQ'],df['PIQ'])

# conclusion: statistic=1.7842019405859857, pvalue=0.08217263818364236
# As pvalue is more than 0.05, so we accept H0, FSIQ and PIQ is same, there is 
# no difference between them.


# ANOVA

#Here are some data on a shell measurement (the length of the anterior
# adductor
# muscle scar, standardized by dividing by length) in the mussel Mytilus 
#trossulus from five locations: Tillamook, Oregon; Newport, Oregon;
# Petersburg,
# Alaska; Magadan, Russia; and Tvarminne, Finland, taken from a much larger 
#data set used in McDonald et al. (1991).
# H0: mean length of different shell is equal
# H1: mean length of different shell is not equal
tillamook = [0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735,0.0659, 
             0.0923, 0.0836]
newport = [0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835,0.0725]
petersburg = [0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]
magadan = [0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764,0.0689]
tvarminne = [0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]

stats.f_oneway(tillamook, newport, petersburg, magadan, tvarminne)
#(7.1210194716424473, 0.00028122423145345439)
# conclusion: As p-value is less than 0.05, so we reject H0 and accept H1.
# this mean mean length of different shell is not equal.

##############################################################################
##############################################################################
#Regression model using dataframe

import pandas as pd
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

boston_data=datasets.load_boston()

print(boston_data.DESCR)

# var = 13
# obs = 506
# target = median value

boston_data.keys()
# ['data', 'target', 'feature_names', 'DESCR', 'filename']

X=pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
Y=pd.DataFrame(boston_data.target,columns=['MEDV'])

print(X.head())
print(Y.head())

print(X.shape)
print(Y.shape)

print(X.isnull().sum())
print(Y.isnull().sum())

# =============================================================================
# Steps to build regression model
# 1. Extraction
# 2. Identify X(feature) and Y(target)
# 3. Data Cleaning (missing value check, 
# 		  outlier check,
# 		  Identification variable check)
# 4. Split X and Y into Train(70%) and Test(30%)
# 5. Build model using train data
# 6. Predict for Test using model
# 7. Validate model using Test data and predicted values
# 
# =============================================================================
# train test split
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.30,random_state=10)
#building model
model=LinearRegression().fit(train_x,train_y)
pred_y=model.predict(test_x)
# model validation
print(mean_squared_error(test_y,pred_y))
print(model.score(train_x,train_y))

# MSE = 29.35
# R-Squared = 0.74

# MSE (Mean Squared Error) - is used to give average error in 
# predicting target value from derived model
# lower error is more preferred.( close to 0)

# R-Squared - is used to give percentage variation in target
# variable explained by derived model. it lies between 0 to 1
# Higer value is more preferred
# R-squared > 0.7 - Good fit model
# R-squared > 0.85 - Best fit model
# R-squared < 0.5 - poor fit model
# model will be Y=W*x+b

# Another method using OLS
import statsmodels.api as sm
train_x=sm.add_constant(train_x)
model2=sm.OLS(train_y,train_x).fit()
# this will give us significance of each variables in model
print(model2.summary())

pred_y2=model2.predict(test_x)
print(mean_squared_error(test_y,pred_y2))
#  R-Squared = 0.964
#  MSE = 30.40
# in OLS method - intercept is not considered by default.
# model will be Y=W*x

##############################################################################
##############################################################################
# Logistic Regression - is used to predict target value which are
# binary. 
# Binary means 0 and 1
# logistic regression will be in form of
# logit(p)= weight*X+bias
# logit(p) = log(p/1-p)
# p = probability of success

# assumption of logistic regression 
# 1. Need not be in linear relation
# 2. Need not be multivariate normal (ordinal or nominal)
# 3. Need not be homoscedascity ( residual can have different variation)
# 4. can have multicollinearity
# 5. Need large sample size ( n > 500) 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#extract titanic.csv file
# find missing values

# 12-4 = 8
# 891 - 2 = 889
# 889,8

titanic=pd.read_csv("C:/Users\Alok/Desktop/Data Science Workshop/Alliance Data/Day 2/Data Files/titanic.csv")
titanic.shape
titanic.head()

titanic.isnull().sum()

titanic.columns
# 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'


titanic_2=titanic.drop(['PassengerId',
                        'Name','Ticket','Cabin'],axis=1)

titanic_2['Age']=titanic_2['Age'].replace(np.nan,titanic_2['Age'].median())
titanic_3=titanic_2.dropna()

titanic_3.shape # 889,8

# we have two character value column - Sex and Embarked
  # we have to convert these two dummy variable.
# dummy variables are numeric conversion of character values
# if you have two unique value in character column, then you will be one
# dummy variable
# no of dummy variable = n-1 where n is no of unique value of character var

# example
# sex      Male
# Female    0
# Male      1

gender = pd.get_dummies(titanic_3['Sex'],drop_first=True)
embark = pd.get_dummies(titanic_3['Embarked'],drop_first=True)

# dropping character columns
titanic_4 = titanic_3.drop(['Sex','Embarked'],axis=1)
titanic_4 = pd.concat([titanic_4,gender,embark],axis=1)
titanic_4.shape
titanic_4.columns

Y = titanic_4.iloc[:,0]
X = titanic_4.iloc[:,1:]

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=10)

logistic_model=LogisticRegression().fit(train_x,train_y)

pred_survive=logistic_model.predict(test_x)

# model validation
print(confusion_matrix(test_y,pred_survive))
print(accuracy_score(test_y,pred_survive)) # c= 0.8 means good fit model
print(classification_report(test_y,pred_survive))

# confusion matrix - is matrix build on test data and predicted values.
# it is used to compare predicted value with actual values
#             0    1  predicted
# actual 0 [[150  19]
#        1 [ 34  64]]

# accuracy_score - is percentage of correctly predicted values.
# (150+64)/(150+19+34+64)

# classification_reports - gives precission and recall for each value of
# target variable
# precission - percentage of correctly predicted value out of total prediction
# recall - percentage of correactly predicted value out of total actual
# F1-score - Harmonic mean of precission and recall

# you can treat accuracy_score as r-squared and if it is above 0.7, then
# model is accepted.

##############################################################################
##############################################################################
#Lasso Regression & Ridge Regression
#Importing the required packages to perform this task
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
#Step2:-I have extracted the data from the path
diamond=pd.read_csv("C:/Users\Alok/Desktop/Data Science Workshop/Alliance Data/Day 2/Data Files/diamonds.csv")
diamond.shape  # 53940,11
diamond.columns
#After Extracting we observed that there are 53940 rows and 11 columns
diamond.drop(['Unnamed: 0','table','depth'],inplace=True,axis=1)
#When inplace = True is used, it performs operation on data and save result to 
# same dataset 

diamond.shape
#Now,after dropping three variable dataset contains 53940 rows*8 columns

diamond.isnull().sum()
#No Missing Value found

# preparing heat map
diamond_corr=diamond.corr()
sns.heatmap(diamond_corr)

# Print unique values of text features
print(diamond.cut.unique())
print(diamond.clarity.unique())
print(diamond.color.unique())

#Tansforming Categorial Value to Numerical value
# Import label encoder
from sklearn.preprocessing import LabelEncoder
categorical_features = ['cut', 'color', 'clarity']
le = LabelEncoder()

# Convert the variables to numerical
for i in range(3):
    new = le.fit_transform(diamond[categorical_features[i]])
    diamond[categorical_features[i]] = new
diamond.head()

# Create features and target matrixes
X = diamond[['carat','x', 'y', 'z', 'clarity', 'cut', 'color']]
y = diamond[['price']]

###SCALING  THE DATA
# Import StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(y)
y=scaler.transform(y)


# =============================================================================
#So, we will split the data into train and test sets, 
# build Ridge and Lasso, and choose the regularization parameter with the help
# of GridSearch.
# For that, we have to define the set of parameters for GridSearch. 
# In this case,the models with the highest R-squared score will give us the 
#best parameters.
# =============================================================================
# Make necessary imports, split data into training and test sets, and choose a
# set of parameters 
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=101)
parameters = {'alpha': np.concatenate((np.arange(0.1,2,0.1), 
                                       np.arange(2, 5, 0.5), 
                                       np.arange(5, 26, 1)))}
# building lasso model
lasso = linear_model.Lasso()
gridlasso = GridSearchCV(lasso, parameters, scoring ='r2')
gridlasso.fit(X_train, y_train)
# Fit models and print the best parameters, R-squared scores, MSE, and 
#coefficients for Lasso Regression


# lasso lambda value 
print("lasso best parameters:", gridlasso.best_params_)
print("R-Squared score:", gridlasso.score(X_test, y_test))   
#R-Squared score: 0.7873154467246168
print("MSE:", mean_squared_error(y_test, gridlasso.predict(X_test)))  
#MSE: 0.21231853566115141
print("Estimatorstimator coef:", gridlasso.best_estimator_.coef_)


#### Building ridge model
ridge = linear_model.Ridge()
gridridge = GridSearchCV(ridge, parameters, scoring ='r2')
gridridge.fit(X_train, y_train)
# Fit models and print the best parameters, R-squared scores, MSE, 
#and coefficients fir Ridge Regression


# Ridge lambda value 
print("ridge best parameters:", gridridge.best_params_)
print("R-Squared score:", gridridge.score(X_test, y_test))  
#R-Squared score: 0.8775879993978104
print("MSE:", mean_squared_error(y_test, gridridge.predict(X_test)))
#MSE: 0.12220133674473616
print("Estimator coef:", gridridge.best_estimator_.coef_)

# =============================================================================
# Now,score raises a little, but with these values of alpha, there is only a 
#small difference.
# =============================================================================

# prediction result and getting final output
pred_y=gridridge.predict(X_test)
final_price=np.abs(scaler.inverse_transform(pred_y))
print(final_price)

