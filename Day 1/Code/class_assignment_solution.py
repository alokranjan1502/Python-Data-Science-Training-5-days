# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:54:42 2019

@author: Alok
"""

# class assignment
#1. convert s1 into nikhiL
#2. convert s1 into niKHil
#3. using s1 and s2 get value "Nikhil Analytics"
s1[:-1]+s1[-1].upper()

s1[:2]+s1[2:4].upper()+s1[4:]
s1.capitalize()+' ' + s2.capitalize()


# find first 3 values of list1
list1[:3]
# find last value of list1
list1[-1]
# find first and last value of list1
print(list1[0], list1[-1])
# extract 20 from list2
list2[1][0]
# extract 30 and 'c' from list2
list2[2]

# Write python program to create a list with below given values
# 100,90,80,70,100,110,120
# give name to list as price_list

# find 5th value of price_list
# find 3rd to 5th index value of price_list
# Create another list called price_list2 with two values in it, first
# value will include first 5 values if price_list and second value will
# include last two values.

price_list=[100,90,80,70,100,110,120]
price_list

price_list[4]
price_list[3:6]

price_list2=[price_list[:5],price_list[-2:]]
price_list2
len(price_list2)


# Class Assignment
# create a list with even number between 20 to 30.
# find 2nd to 4th value of list
# replace 2nd value with 23
# add new value 32 in the last
# sort data in descending order
# add two value 34 and 36 in the last

evenno=list(range(20,32,2))
evenno
#range() - is used to generate a sequence of values from give start value
# and stop value with a step value
# syntax- range(start,stop,step)
# stop value is exclusive
evenno.extend([32,34,36])
print(evenno)
# replacing value in list
evenno[1]=23

#Nested list
Weekly_pricelist=[['mon',100],['tue',110],['wed',120],
                  ['thu',128],['fri',140]]
print(Weekly_pricelist)
# solve below give questions
# find 2nd value of given list
# find 2nd sub value of 2nd elemnet
# replace 128 by 130
# add new value to list ['sat',150]
# delete ['tue',110]

# some extrac function of list
x=[10,11,12,10,13,14,10]
x.count(10) # return number of times values persent in list
y=x.copy()  # create a copy of given list
x.pop(3) # pop up value which is going to be remove from given index
x.clear()  # remove entire list

#class assignment
# Write python function to find new value based on below given condition
# condition        new_value
#  below 10          square
#  10 to 20          square root
#  above 20          as it is

def get_value(x):
    if(x<10):
        new_value=x*x
    elif(x<20):
        new_value=round(math.sqrt(x),2)
    else:
        new_value=x
    return new_value;

get_value(2)
get_value(12)
get_value(22)

get_value(-4)

# class assignment-
# write python code to find square of 1 to 10 except 5
#A. what to do in this question,
#B. we will wait, sir wil give solution    
#C. if any body say i have done, i will copy from him/her
#D. it is very difficult topic

for i in range(1,11):
    if(i==5): continue
    print(i*i)
    
#range()- create sequencial values from given start to stop-1
    
#class assignment
# how many employees have salary above 25k
np.sum(employee['Salary']>25000)

# average salary for employee whose salary is above 25k
np.mean(employee[employee['Salary']>25000]['Salary'])    



data5=pd.read_csv('SampleCSVFile2.csv',encoding='iso 8859-1',header=None)
data5.shape
data5.head()
# to read data from sql or any database-
# you have to create a DSN (Data Source Name)
# In order to extract data from database, we need DSN(Data Source Name)
# we have to first create DSN (in Orgnization, DBA used to create this)

# How to create user DSN
# goto control panel->select system and security->administrative tools
#-> Data SOurce(ODBC)->user DSN->click add-> in popup window
# give name to DSN (it will be used in python code) and server name
# select finish->next->change default database to database which
# you want to extract->next->next->test data source connection
# click ok to close all windows

import pyodbc
myconn=pyodbc.connect('DSN=vishal')
data4=pd.read_sql('select * from security',myconn)
data4.shape
print(data4)




import pandas as pd
import numpy as np
from scipy.stats import pearsonr


data1=pd.read_excel("D:/Python/Python Class notes/Part 2/Class 8/Assignments/Data1.xlsx")

data1.shape  # 406,9

data1.columns
 #['MPG', 'CYLINDERS', 'DISPLACEMENTS', 'HORSEPOWER', 'WEIGHT',
 #      'ACCELERATION', 'MODEL YEAR', 'ORIGIN', 'CAR NAME']
 
 
np.corrcoef(data1['MPG'],data1['WEIGHT'])
# nan - because there is missing values in MPG

pearsonr(data1['MPG'],data1['WEIGHT'])
# nan - because there is missing values in MPG

data1['MPG'].corr(data1['WEIGHT'])
# r=-0.83 - negative strong correlation, this means heavier vehicles have less MPG

# Note - corrcoef() and pearsonr() does not work when you have missing value data
# nan - is taken as missing value. please check MPG column data to understand

# we have to go for missing value treatment i.e replace missing value with median
# before performing any sort of analysis.

data1_new=data1.fillna(data1.median())
np.corrcoef(data1_new['MPG'],data1_new['WEIGHT'])
pearsonr(data1_new['MPG'],data1_new['WEIGHT'])
# r= -0.8239854383071897, pvalue=1.0213856188321698e-101

# pvalue < 0.05, so we reject H0 and accept H1. Correlation between MPG and WEIGHT
# is accepted.


data1A=data1[['MPG', 'HORSEPOWER', 'WEIGHT','ACCELERATION']]

data1A.corr()

import seaborn as sns

sns.heatmap(data1A.corr())

# prepare heatmap for data2 correlation 

# identify strong correlation ? 
# wind and arrow , height and slice - has positive strong correlation

# Assignment
# class assignment
location="D:/Python/Python Class notes/Part 2/Class 9/"
dataset = pd.read_table(location+'pima-indians-diabetes.data.txt', 
                    sep=",",skiprows=12,header=None)
dataset.shape #768,9

# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, np.NaN)

dataset.isnull().sum()

# perfrom following on given dataset
# 1. variable 1 - replace missing value with previous value - df.fillna(method='bfill')
# 2. variable 2 - replace missing value with 40             - df.replace(np.nan,40)
# 3. variable 3 - replace missing value with median         - df.fillna(df.median())
# 4. variable 4 - drop this variable                        - df.drop(['varname'],axis=1)
# 5. variable 5 - drop missing values observations          - df.dropna()

# Assignment -1
 # conclusion - pvalue = 0.07 as pvalue is more than 0.05, so we accept H0
# there is no association between Shift and Quality 
 

 # pvalue = 0.007 as pvalue is less than 0.05, so we reject H0
 # xsq = 20.90 
 # xsq from table = 15.9
# xsq calculated is more than xsq from table, we reject H0

 #assignment -1
# =============================================================================
# 	Perfect	   Satisfactory	Defective
# Shift1	106	     124		1
# Shift2	67	      85		1
# Shift3	37	      72		3
# =============================================================================
# H0: shift and quality is not associated
# H1: shift and quality have association
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
a = np.array([106,67,37,124,85,72,1,1,3])
b=a.reshape(3,3)
d=pd.DataFrame(np.transpose(b),columns=['Perfect',' Satisfactory','Defective'],
               index=['Shift1','Shift2','Shift3'])
d
#or
import pandas as pd
assignment_1=pd.read_table("D:/Python/Python Class notes/Part 2/Class 10/Assignment/Assignment 1.txt",
                           skiprows=3,delim_whitespace=True)

assignment_1.shape # 3,3


xsq,pvalue,dof,expected=chi2_contingency(d)
print(xsq,pvalue) # 8.646695992462913 0.07056326693766583
# as pvalue is more than 0.05, so there is no association between 
# shift and quality.

# assignment -2
# =============================================================================
# Acme Toy Company prints baseball cards. The company claims that
# 30% of the cards are rookies, 60% veterans but not All-Stars,
# and 10% are veteran All-Stars.
# 
# Suppose a random sample of 100 cards has 50 rookies, 45 veterans,
# and 5 All-Stars. Is this consistent with Acme's claim? 
# Use a 0.05 level of significance.
# Also write you null and alternative hypothesis.
# 
# =============================================================================
# H0: given sample is consistent with company claim
# H1: given sample is not consistent with company claim


#########################################################################################
#########################################################################################

