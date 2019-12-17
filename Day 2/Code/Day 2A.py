# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 07:33:26 2019

@author: Alok
"""

##############################################################################
##############################################################################
# lambda function
mylist=[10,20,30,40,50]
# increment all values of mylist by 5

newlist=[]
for i in mylist:
    newlist.append(i+5)
    
print(newlist)
# lambda function - is used to perform calculation and return value just like
# user defined function. But it does not create any function.
# syntax - lambda args:calculation

f1=lambda x:x+5

print(f1(10))

# map function - is used to perform calculation to each value of given list,
# using lambda function
# syntax - map(lambda args:calculation, list_name)

newlist=list(map(lambda x:x+5,mylist))
print(newlist)

# filter function - is used to fiter list value based on some condition
# using lambda function
#syntax - filter(lambda args:condition , list_name)

above_20=list(filter(lambda x:x>20,mylist))
print(above_20)

# class assignment
newlist=[2,3,5,6,9,10]

# create new list by taking only odd values for newlist

odd_values=list(filter(lambda x:x%2,newlist))
print(odd_values)

even_values=list(filter(lambda x:x%2==0,newlist))
print(even_values)

# reduce function() - is used to reduce list value to single 
#value by using lambda function.
# syntax - reduce(lambda args:calculation , list_name)

from functools import reduce
total=reduce(lambda x,y:x+y,mylist)
print(total)
# here we are finding cumulative value of mylist.
# in first iteration x and y will take first and second value of mylist
# but in second iteration x will take result of first iteration and y will take 
# third value and so on
# x  y   total
# 10 20  30
# 30 30  60
# 60 40  100
# 100 50 150

# you can perform above task without using these functions,
# You can do these by list comprehension(using for loop)
# its own choice how to do give task

# list comprehension

newlist2=[x+5 for x in mylist]
print(newlist2)

odd_num=[x for x in newlist if x%2]
print(odd_num)


# missing value detection and treatment

# when you extract data from different sources, you will get
# nan as welll if there is  no data present in source location.

# nan - Not a Number and it is taken as missing value.

# if you are getting values such as "?" , .(dot) or blank then
# these are not taken as missing value, you need to replace all
# these value by nan or NaN

# How to find missing values in data ?

# 1. np.isnan(data['varname']).sum() - gives count of missing value
# in given variable

# 2. data.isnull().sum()  - gives count of missing values of all
# variables of given dataframe

# Your function will return missing value if you have missing 
# value data. so it is very important that you sould go for 
#treatment

# missing value treatment
# 1. drop missing value - data.dropna(axis=0)  - is used 
#for large dataset all rows which has missing value will be dropped

# 2. replace missing value with previous value or next value - 
#used for character values
  # data.fillna(method="bfill")
  # data.fillna(method="pad")

# 3. replace missing value with given value -
# used as per business logic
  # data.replace(np.nan,value)

# 4. replace missing value with median value - 
#used for numeric values
  # data.fillna(data.median())

import numpy as np
import pandas as pd

data=pd.DataFrame(np.random.rand(10,4),columns=list('ABCD'))
data

data.iloc[2:3,0]=np.nan
data.iloc[3:6,1]=np.nan
data.iloc[4:9,2]=np.nan
data

# find number of missing values
np.isnan(data['B']).sum()
data.isnull().sum()  # gives missing values for all columns

# 1. drop missing values observations
data.dropna()

# 2. replace missing with previous value
data.fillna(method="bfill")

# 3. replace missing with 0
data.replace(np.nan,0)

# 4. replace missing value with median
data.fillna(data.median)

# Assignment
# class assignment
# extract pima_indians_diabetes.csv file and perform below task
# 1. variable 1 - replace missing value with previous value - df.fillna(method='bfill')
# 2. variable 2 - replace missing value with 40             - df.replace(np.nan,40)
# 3. variable 3 - replace missing value with median         - df.fillna(df.median())
# 4. variable 4 - drop this variable                        - df.drop(['varname'],axis=1)
# 5. variable 5 - drop missing values observations          - df.dropna()


#Outlier values - detection and treatment

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(10)
# seed - is used to re-produce same result again and again.
# when ever we use a random function to generate random value, every time
# we get new values, that can generate different results. 
# in case you want same result every time, then set seed value to any positive
# integer values
array1 = np.random.normal(100, 10, 200)
array2 = np.random.normal(90, 20, 200)
data = [array1, array2]
data
len(data[0])
#method - 1
#High_outlier - Q3+1.5*IQR
#Low_outlier  - Q1-1.5*IQR
#Outlier will be replaced with median
plt.boxplot(data)
res = plt.boxplot(data)

array1_outlier= res["fliers"][0].get_data()[1]
array1_outlier
# outier values are 
# array([ 70.20403229,  77.04896671, 123 .94703665, 124.67651056,
#       124.04325606, 124.65325082])
array2_outlier= res["fliers"][1].get_data()[1]
array2_outlier
#array([143.59820616])
#res['boxes'][0].get_data()[1]


#method - 2  - when data is huge ( big data - millions of observations)
# formula - mean+-3*std - values above or below this formula 
# are taken as outlier.
df = pd.DataFrame(array1,columns={'Data'})
df.describe()
df[(np.abs(df.Data-df.Data.mean())>(3*df.Data.std()))]
# outlier values are
#  70.204032

# replacing outlier values with median 
df=df.replace(70.204032,df.median())

##############################################################################
##############################################################################
#Chi- Square test - is used to find association between two
#categorial values such as gender,type,grade,like/dislike etc

# xsq = sum((expected-observed)^2/expected)

# chi-square test assumption
# H0: there is no association between given variable(xsq=0)
# H1: there is association between given variable(xsq not=0)

# Accept H0 when pvalue is more than or equal to 0.05, else reject H0.
# reject H0 means you have to accept H1.

# chi-square test is also called as non-parameteric test because it
# does not use mean or std

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

data=pd.read_table("Hair_Eye_Color.txt"
                   ,delim_whitespace=True)
data.shape  # 27,3
data.columns

# find association between Hair and Eye color?
# first create contingency table using hair and eye

contigency_table=pd.pivot_table(data,index=['Hair'],
                                columns=['Eye'],
                                values=['n'],
                                aggfunc=np.sum)
contigency_table
# replace missing value with zero
contigency_table2=contigency_table.replace(np.nan,0)
contigency_table2
# performing chi-square test
xsq,pvalue,dof,expected=chi2_contingency(contigency_table2)
# dof - degree of freedom  = (r-1)*(c-1)
# expected = rt*ct/total_obs
# r= no of row
# c=no of columns
# rt = row total
# ct = column total
(6.0+51.0+69.0+68.0+28.0)(6.0+16.0+0.0)/total


print(xsq)
print(pvalue)
print(dof)
print(expected)


# Goodness of fit test - is used to test population distribution with
 # expected distribution.
# here you will get sample data with expected distribution

# Suppose, A company want to know his employees interested to go GYM or not
 # If there is Majority, they invest in setting Gym within compus.
 # sample = [100,140,40,80]
 # expected = [0.30,0.50,0.1,0.1]
 
 # H0: given sample is consistent with expected value
 # H1: given sample is not consistent with expected value
 
from scipy import stats
import numpy as np
sample=[100,140,40,80]
expected=[0.30,0.5,0.1,0.1]
 
sample_per=sample/np.sum(sample)
 
xsq,pvalue=stats.chisquare(f_obs=sample_per,f_exp=expected)
print(xsq)   # 0.17
print(pvalue) # 0.98
 
# As pvalue is more than 0.05, so we accept H0. sample is consistent with 
# expected value
