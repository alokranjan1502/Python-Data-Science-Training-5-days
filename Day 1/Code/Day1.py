# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:12:29 2019

@author: Alok
"""

#Basic of programming and file handling
#to print 
print("Hello, I am from Nikhil Analytics")
# ctrl+enter - is used to execute program

#to print assigned variables
name="ALOK"
learn="PYTHON"

print(name , " is learning ", learn)

Age=50

print(name," is " , Age , "year old")

x=100

x/2

y=x/2
y
transaction=10000
tax=transaction*(10/100)
print(tax)










print(x,y)

x="abc"

#assigning variables and calculating & printing the sum
a=10
b=20
c=30

total=a+b+c
print(total)

###############################################################################
###############################################################################
#Data Types in Python and their usages 
# =============================================================================
# Data Type - in python we have 5 different types of values.
# These values are called as data types
# 1. integer - no decimal value
# 2. float   - numeric with decimal
# 3. string  - any value with quotation(' ' , " ", """  """)
# 4. boolean - True,False
# 5. complex - a+bj (a-real value, b=imaginary value)
# 
# =============================================================================

# =============================================================================
# in order to store multiple values,we use three different Objects
# 1. list       - [] , list is mutable
# 2. tuple      - () , tuple is immutable
# 3. dictionary - {} , dictionary is mutable
# 
# =============================================================================

i1=10000; i2=20000

type(i1)

f1=10.5 ; f2=20.0
type(f1)
type(f2)

# conversion 
#int()   - return int value
#float() - return float value

int(f1)
float(i1)

# String values
s1='nikhil'
s2='ANALYTICS'
# note - string values are taken as string list. each character is
# taken as individual value and these are stored at different index
s1[0]
s2[-1]

s1.upper() # return upper case of s1 values
s2.lower() # return lower case of s2 values

# convert s1 into 'Nikhil'
s1[0].upper()+s1[1:]

# class assignment
#1. convert s1 into nikhiL
#2. convert s1 into niKHil
#3. using s1 and s2 get value "Nikhil Analytics"



# you can store all sort of special character to string
s3='alok@gmail.com'
s4='A101'
s5='101'

i3=101
x=5*2

#concatemation
s5+str(10)
i3+x

# Boolean value - True and False
b1=True
b2=10>20

#Boolean operations
b1 & b2 # b1 and b2  - give true if both are true else false
b1 | b2 # b1 or b2   - any one is true return true
b1!=True # not equal to

# complex data - a+bj ( a= real value, b=imaginary value, j^2=-1)
c1=1+2j
c2=2+3j
type(c1)

c1+c2  # 3+5j
c1*c2  # -4+7j
# 1+2j * 2+3j = 1*2 +2j*2 +1*3j +2j*3j = 2+4j+3j+6j^2=2+7j-6=-4+7j
c1.real
c1.imag

#########################################################################################
#########################################################################################
# When you want to store multiple values, you need to use either
# list, tuple or dictionary
# =============================================================================
# list - we know it
# tuple - is a type of list but you cannot modify or change it values
# dictionary - is used to store values in form key:value pair
# 
# =============================================================================
my_list=[10,20,30]
my_tuple=(10,20,30)
my_dict={'a':10,'b':20,'c':30}

# extract data 20 from list,tuple and dictionary
# replace 20 by 25

my_list[1]
my_tuple[1]
my_dict['b']

my_list[1]=25
my_tuple[1]= 25  # give error, because any change in tuple is not allowed
# tuple can be changed when you re create it.
my_dict['b']=25

print(my_list)
print(my_dict)

my_dict.keys() # return all keys of dictionary
my_dict.values() # return all values of dictionary
# you can access values of dictionary only with the help of keys()

# adding values to dictionary
my_dict['d']=40
my_dict

# removing values from dictionary
del my_dict['b']
my_dict

#########################################################################################
#########################################################################################
#List in Python
# List - is one dimensional array to store multiple valus of different types.
# different types means you can store character,numeric or any other types
# of values in single list.
# to create list use [] , other brackets will not create list.

# As python is case sensitive as well as space sensitive, so you have to be
# careful while writing scripts(programs).

list1=[10,'a',20,'b',30,'c']
print(list1)

print(len(list1))  # len() - gives count of values in given list
print(type(list1)) # type() - gives data type of given variable

list2=[[10,'a'],[20,'b'],[30,'c']] # nested list
print(list2)
print(len(list2))

list3=[[10,'a',20,'b'],[30,'c']] # nested list
print(list3)

# extract 
list1[2]
list1[-4]

# to get more than two values, we use list slicing
# listname[star:stop]  start and stop repersent index position
# here start in inclusive and stop is exclusive

list1[2:4]  # return 2nd and 3rd index value

# listname[start:]  from given start index to till last
list1[3:]

# listname[:stop]  from 0 index to stop-1 index
list1[:3]

# listname[:]  all values from given list
list1[:]

# class assignment
# find first 3 values of list1
# find last value of list1
# find first and last value of list1
# extract 20 from list2
# extract 30 and 'c' from list2


# Write python program to create a list with below given values
# 100,90,80,70,100,110,120
# give name to list as price_list

# find 5th value of price_list
# find 3rd to 5th index value of price_list

# Create another list called price_list2 with two values in 
#it,
# first value will include first 5 values if price_list and 
#second value will include last two values.

price_list=[100,90,80,70,100,110,120]
price_list

# List Manipulation
#1. Adding new value to list
price_list.append(50)
print(price_list)

#2. remove value from list
price_list.remove(70)
print(price_list)

#3. find position of given value
price_list.index(120)
price_list.index(130)

#4. insert new value to specific position
price_list.insert(3,130) # insert 130 at 3rd index
print(price_list)

#5. reverse values of list
price_list.reverse()
 # re arrange data in order of last to first
print(price_list)

#6. sort values of list
price_list.sort()  # sort data in ascending order
print(price_list)
##############################################



# Class Assignment
# create a list with even number between 20 to 30.
# find 2nd to 4th value of list
# replace 2nd value with 23
# add new value 32 in the last
# sort data in descending order
# add two value 34 and 36 in the last


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






#########################################################################################
#########################################################################################
# function - are used to perform calculation and
# return values
# you can use system defined function which 
#are object based 
# such as list_name.append() , list_name.remove(), 
#list_name.sort() etc
# you can create your own function which 
#can be used for object or variables.
# ALl user defined function are temporary which means you
# can use these function for only one session.
# for next session you have to create again.

# How to create function - 
#def function_name(args):
#    python statements;
#    return variables;

# in order to use function, you need to call it with values
#function_name(values/variables)

# Note - please notice space given in next line of def statement
# these space are called as indent, if you keep these space you
# statement is considered as part of function and if you remove
# it (even one space) will be taken as out of function and you 
# will get error or wrong result.

# write a python function to increment a given value with x 
def increment_by_x(value,x):
    final=value+x
    return final; 

increment_by_x(1000,20)

increment_by_x(2000,30)



# write a python function to find value after
# 10% discount to user 
# given value.





def get_finalValue(user_value,discount=10):
    discount_amt = user_value*discount/100
    final_amt = user_value-discount_amt
    return final_amt;

get_finalValue(100)

get_finalValue(100,12)
get_finalValue(100,15)
get_finalValue(100,5)

# here discount value is taken as 10 by default, if you give value
# then it will take your values otherwise 10 by default

# scope of variables
# local  - variable created within function.
# it is used only within function
# global - variable created outside function. 
#it can be used any where.

#print(discount_amt)

# can global variable be used within function ?
# yes, you can use global variable within function,
# but if you change global
# variable value, it will change function result also.

inc = 10
def get_increment(x):
    y = x*(1+(inc/100))
    y=round(y,2)
    return y;

# 10% increment in 100
get_increment(100)

inc=20
get_increment(100) # 20% increment in 100

round(10.782563,3)

#########################################################################################
#########################################################################################

# Decision Making statements
# if
# while
# for

# if statement - is used to perfrom calculation when
# given condition is true
#type-1  if (condition): action1
#type-2  if (condition):
#           action1
#           action2
#        other python statement

#type-3 if (condition1):
#           action1
#       elif (condition2):
#           action2
#       else:
#           action3






# write python function to find discount percentage 
#given to customer
# based on below offer.
# buy 1 - 40% off
# buy 2 - 50% off

def get_discount(qty):
    if (qty==1):
        discount=40
    elif (qty==2):
        discount=50
    else:
        discount="Not defined"
    return discount;

get_discount(2)
get_discount(3)
get_discount(1)


# write python code to find square and square root
# of user given value if it is below 10.














import math
def get_sq_sqroot(x):
    sq=x;sqroot=x
    if(x<10):
        sq=x*x
        sqroot=round(math.sqrt(x),2)
    else:
        print("Given x is more than 10")
        print("sq and sqroot is assign to x itself")
    return sq,sqroot;
    

get_sq_sqroot(5)

x,y=get_sq_sqroot(15)
print(x,y)

get_sq_sqroot(15)

#class assignment
# Write python function to find new value based on 
#below given condition
# condition        new_value
#  below 10          square
#  10 to 20          square root
#  above 20          as it is


# =============================================================================
# While Loop - is used to execute set of 
#instructions while condition is true.
# syntax - 
# while(condition):
#     python statement1
# else:
#     python statement2
# 
# Here Python statement1 will execute when given 
#condition is true, and
# it will keep on execute while condition is true.
# If condition is false
# will stop and go to else. 
#python statement2 will be execute only one time
# when condition is false. Else part is optional.
#     
# =============================================================================

# Write python function to find number of time 5 
#need to be added to value x to get 50. 










    
def get_50(x):
    count=0
    while(x<50):
        count+=1
        x+=5
    else:
        print("Count is " , count)
    
get_50(50)
get_50(40)


# =============================================================================
# for loop - is used to execute set of python 
#statements one time for each value in given collection.
#  syntax
#  for i in list:
#      python statements
#      
#      
# =============================================================================

my_list=[10,20,30,40]

# write python code to increment all 
#values of given list by 10%

new_list=[]
for i in my_list:
    new_list.append(i*(1+(10/100)))

print(new_list)

# Write python function to find number of years it
# will be accumulate 200000
# if i deposit 5000/-per month to my bank account.
# If bank interest rate is
# 4% annum and interest compounded monthly.

def get_required_year(permonth,roi,req_amt=200000):
    total=0; month=0
    while(total<req_amt):
        total+=permonth
        total+=total*(roi/1200)
        month+=1
    req_year=round(month/12,2)
    return req_year;

get_required_year(5000,4)

get_required_year(5000,4,500000)

get_required_year(10000,4,500000)


# write python function to find number of years it will to get double of 
# money deposited in first year. (one time deposit)
# if interest rate is 8.5% per annum and interest compounded yearly

    
def get_required_year2(amt,roi):
    total=amt; year=0
    req_amt=2*amt
    while(total<req_amt):
        total+=total*(roi/100)
        year+=1
    return year;

get_required_year2(200000,8.5)


# when to use for loop and when to use while ? 
# you can use any of these two for all cases. But it will be good
# if you use for loop for given set of interval and while
# interval or range is not given

# case - 1 A boy is saving 7/- per week. 
#How much he wil accumulate  in one year

# case - 2 Same boy of case-1,
# wants to know how manys days it will take him to save 500/-



# A person want to make 500000/- and he is saving 1000/-
# per month
# in bank. Bank gives 4% interest per annum. If interest is
# compounded monthly. Then how many month it will take.

permonth=1000
interest=4
reqamt=500000
total=0
month=0
while(total<=reqamt):
    total+=permonth
    total+=total*(interest/12)/100
    month+=1
print("Total amount:",total," Required month:", month)

295/12
198/12
136/12

def Req_month(permonth,interest,reqamt):
    total=0
    month=0
    while(total<=reqamt):
        total+=permonth
        total+=total*(interest/12)/100
        month+=1
    return [total,month]

Req_month(1000,4,500000)

# =============================================================================
# Loop control statement 
# 1. Break - is used to stop loop statements
# 2. Continue - is used to stop current iteration and
# continue for next iteration
# 3. Pass - is used to pass execution to next line,
# doesnot do any special work
#               
# =============================================================================
    
for i in "PYTHON":
    print(i)

for i in "PYTHON":
    if(i=="T"): break
    print(i)
    
for i in "PYTHON":
    if(i=="T"): continue
    print(i)
    
for i in "PYTHON":
    if(i=="T"): pass
    print(i)
 

#########################################################################################
#########################################################################################
#session-2
# numpy 
    
#Package - package are compressed files which 
#include multiple functions
# and objects. To use function of package, 
#we have to import it first
# in our python session.
# To get package you have to install it through Anaconda 
#environment
# to use function of package in code, execute below code
# import packagename as alias
# 
# import numpy as np
# 
# to use function package use below code
# package alias.functionname
# 
# np.sum()
# np.abs
# np.sqrt
# 
# some packages are very big, so we take only few function
# of it.
# from package_name import function_name
# 
# from numpy import std
import numpy as np
number_array2=np.array([[23,56,23,34,56,76],
                        [52,51,63,82,73,58]])
print(number_array2)
number_array2.shape

# find if 56 is present in number_array2

56 in number_array2[0]  # searching in first row
56 in number_array2[1]  # searching in second row

# find position of 56 in number_array2

list(number_array2[0]).index(56)
number_array2.shape

# =============================================================================
# pandas  - is used for general purpose task such as 
#creating table
# ( dataframe or dataset), manipulation, sorting, 
#transposing etc
# 
# =============================================================================

import pandas as pd
# in order to store values in pandas, we have two objects
# Series    - is used to store only single column 
#(one dimensional array)
# DataFrame - is used to store in two dimensional array.
# it can store different types of values.
import pandas as pd
data1=pd.Series([10,20,30,40])
print(data1)
print(type(data1))


employee=pd.DataFrame({
        'Id':range(101,107),
        'Name':['Ravi','Mona','Rabina','Mohan','Ganesh',
                'Raju'],
        'Sex':['M','F','F','M','M','M'],
        'Age':[20,22,24,26,28,30],
        'Salary':[20000,22000,24000,26000,28000,15000]})

# find number of rows and columns in employee dataframe
# find column names of employee dataframe
# find unique value of Sex
# Convert id values to A101,A102,A103....

employee.shape
employee.columns
employee.Sex.unique()

employee.Id=['A'+str(i) for i in employee.Id ] 
employee

# display top 3 rows
employee.head(3) 
 # in case you does not give number, 
 #it will take 5 by default

# display bottom 3 rows
employee.tail(3) 
# in case you does not give number, 
#it will take 5 by default

# extraction
# single column
employee['Name']
employee.Name

# multiple column
employee[['Name','Age']]

# rows
employee[0:3]  # first 3 rows values

# first three name
employee.Name[0:3]
employee.Name.head(3)

employee.iloc[0:3,1]

#employee.ix[0:3,1]

# find salary of Rabina
employee
# find salary of Rabina and Ravi   - 
#this is your google assignment

# find name of person whose salary is lowest
import numpy as np
np.min(employee['Salary'])

employee[employee['Salary']==np.min(employee['Salary'])]['Name']


# data frame manipulation
# adding column
employee['Bonus']=0.30*employee['Salary']
print(employee)

# dropping column
employee2=employee.drop(['Age'],axis=1)

# sorting
employee.sort_values(by=['Salary'])  # ascending order
employee.sort_values(by=['Salary'],ascending=False)  # descending order

# tranpose
employee.T
# group by calculation
employee.groupby(by=['Sex'])['Salary'].mean()
# average salary for male and female

# updating column values
# increment salary by 5%
employee['Salary']=employee['Salary']*1.05

#class assignment
# how many employees have salary above 25k
np.sum(employee['Salary']>25000)

# average salary for employee whose salary is above 25k
np.mean(employee['Salary'][employee['Salary']>25000])


#########################################################################################
#########################################################################################

# Extraction - How to read data into Python, or say fetching data from
# different sources to Python

import pandas as pd
import os   # is used to set path for working directory

# setting path for folder location
os.chdir("C:/Users/Alok/Desktop/Data Science Workshop/Alliance Data/Day 1/Data file")
# note - you need to replace all "\" to "/".
# reason "\" has special meaning in python, to avoid we have to use 
# either "\\" or "/"

# to see all files of current directory
os.listdir()

# to read text file - pd.read_table()
data1=pd.read_table('company.txt',sep="/",header=None)
print(data1.shape)
print(data1)

# to read csv file - pd.read_csv()
data2=pd.read_csv("crabtag.csv")
data2.shape

# to read excel file - pd.read_excel()
data3=pd.read_excel("sales.xlsx")
data3.shape
# excel file can have multiple sheets.
# in order to read sheet2 data, what
# we have to do

data3A=pd.read_excel("sales.xlsx",sheet_name="Sheet1")
data3A.shape


# try to read data sampleCSVFile2.csv and student.txt

#########################################################################################
#########################################################################################
# I received a large dataset from my client, what should i do first?
# we do descriptive statistics on given data.
# Under descriptive statistics we analysize each column or variables with
# different statistical values.
# descriptive statistics is used to find various statistical values which
# help us to understand our data and help in making decision.
#1. Central Tendency - mean,median,mode
#2. dispersion - variance,standard deviation, range,Inter Quartile range
#3. skewness - spreadness of data along with mean
#4. kurtosis - peakedness of distribution

import pandas as pd

sales=pd.read_excel("sales.xlsx")
sales.shape  # 780,3
sales.columns
sales.describe()  # gives descriptive statistics values of all numeric
# variables of given dataframe

# Central Tendency
sales['Sales'].mean()    # 8507.33
sales['Sales'].median()  # 7398.5
sales['Sales'].mode()    # 5082

# mean   - average value ( sum of all vlaues/no of values)
# median - middle most value
# mode   - most frequently occuring values 
# why we need this ?
# this will help us to understand our customer requirement and their
# demographical information.

# dispersion
sales['Sales'].var()
sales['Sales'].std()
sales['Sales'].max()-sales['Sales'].min()
sales['Sales'].quantile([0.25,0.5,0.75])

#variance(var) - average difference between observation and its mean value
#  = sum((obs-mean)^2)/n-1
# standard deviation = sqrt(var)
# quantile - divide data into four equal parts after sorting into ascending
# order 
# why we need this ?
# this will help us to understand range of our potential customer and 
# their requirement.
# in case of large values, we have large variation in customer requirement,
# which will be difficult to meet.

# skewness - spreadness of distribution
# if data has equal distribution above and below mean, then skewness is zero
# more above mean - + skewness or right skewed
# more below mean - - skewness or left skewed

# kurtosis - peaked of distribution
# if data has more peaked value, then it has positive kurtosis
# if data has flat value, then it has negative kurtosis

sales['Sales'].skew()  # 0.23  - more values above mean
sales['Sales'].kurt()  # -1.6  - flat curve, wide spread curve

import seaborn as sns
sns.distplot(sales['Sales'],hist=True,kde=True)

# why we need this?
# Skewness and kurtosis will help us to understand our customer
# purchase behaviour. Skewness will tell us cusotmer is buying 
# above mean product or below mean product. 
# kurtosis will help us to understand customer income range, and 
# their needs.


#########################################################################################
#########################################################################################
# Finding relation between variables
# In case of numeric variable, we use correlation to find relation between them.
# numeric variables - age,height,weight,salary,distance,quantity etc

# In case of character variables, we use chi-square test to find association
# between them. character variable ( categorial variables) - gender,type,group,
# grade, rank etc

# in case of one numeric and one character variable, we use ttest to check 
# significance of variables.

# correlation - is used to find relation between two numeric variables.
# it is measures by correlation coefficient (denoted by r)
# r(x,y) = cov(x,y)/std_x*std_y     - pearson correlation
# correlation of x and y = covariance of x and y / std of x * std of y
# covariance = sum((x-xbar)*(y-ybar))/n-1
# x - observations of x, xbar - mean of x
# y - observation of y, ybar - mean of y
# n - total observations

# correlation test- this is used to test calcuated correlation coefficient is
# valid for population or not.
# H0: (our assumption) - correlation is equal to zero.
# H1: (opposite of assumption) - correlation is not equal to zero.
# accept H0 when pvalue is greater than or equal to 0.05
# reject H0 when pvalue is less than 0.05, and accept H1.
# pvalue (probability value) it will be given system function.

#  Value of r will lies between -1 to 1.
# -ve value - negative correlation - both variables moves in opposite direction
# +ve value - positive correlation - both variables moves in same directions
# zero - no correlation.

# height and weight     - positive correlation
# age and learning rate - negative correlation

# based of value of r, we can define correaltion in three types
# |r| 
# 0-0.3     - weak correlation
# 0.3-0.5   - moderate correlation
# 0.5-1.0   - strong correlation

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

x = np.random.randint(0,100,1000)
y = x+np.random.normal(0,10,1000)

np.corrcoef(x,y)
# 0.94 - positive strong correlation

pearsonr(x,y)
# r = 0.94, p=0.0  - as pvalue is below 0.05 so we reject H0 and
# accept H1. correlation is not equal to zero

data = pd.DataFrame({'x':x,'y':y})
data['x'].corr(data['y'])
# 0.94

# we can correaltion graphically as well.
import matplotlib.pyplot as plt
plt.scatter(x,y)

z = 100-x+np.random.normal(0,5,1000)
plt.scatter(x,z)
np.corrcoef(x,z)
# -0.987  - negative strong correlation

data1 = pd.DataFrame({'x':x,'y':y,'z':z})
#data=pd.concat([data,pd.Series(z)],axis=1)
#data.columns=list('xyz')
data1.corr()  # correlation matrix

# heat map of correaltion matrix
import seaborn as sns
sns.heatmap(data1.corr())


#########################################################################################
#########################################################################################
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

# reduce function() - is used to reduce list value to single value by using
# lambda function.
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
# nan as welll if there is  no data persent in source location.

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
# value data. so it is very important that you sould go for treatment

# missing value treatment
# 1. drop missing value - data.dropna(axis=0)  - is used for large dataset
# all rows which has missing value will be dropped

# 2. replace missing value with previous value or next value - used for character values
  # data.fillna(method="bfill")
  # data.fillna(method="pad")

# 3. replace missing value with given value - used as per business logic
  # data.replace(np.nan,value)

# 4. replace missing value with median value - used for numeric values
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
dataset = pd.read_table('pima-indians-diabetes.data.txt', 
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
# array([ 70.20403229,  77.04896671, 123.94703665, 124.67651056,
#       124.04325606, 124.65325082])
array2_outlier= res["fliers"][1].get_data()[1]
array2_outlier

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
#########################################################################################
#########################################################################################
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
