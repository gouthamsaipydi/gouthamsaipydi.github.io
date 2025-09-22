---
layout: post
title: "7 Pandas Methods everybody should know"
date: 2025-09-22
tags: []
categories: [Pandas]
description: "A short compilation of the most used pandas methods"
featured: true
thumbnail: 
---

Lo and Behold, here i am back with the 7 most important methods in pandas that every data scientist should be aware of. 
    
Note: I'll be skipping over the basic methods like `head()` and `info()`, coz they are the best. 

1) `describe()`
	- This method can be applied on both Data Frames and Series; its a quick way to breeze through the descriptive statistics of the columns involved in the dataset. 
	- The 5 point summary of data can be a good starting point for getting an idea about the data distribution. The Five Numbers are Minimum, Q1(1st Quartile), Median, Q3(3rd Quartile) and Maximum
	- Along with that, the method also provides the frequency(count),mean and standard deviation to better understand the spread.
	- For object(string) columns; the above method provides us with the count, number of unique objects, the most frequent element and also its frequency.

	```python
	import pandas as pd 
	df=pd.DataFrame({'Age':[14,35,25,36,19],'Height':[144,157,178,181,171]})
	df.describe()
	#output
	#>>> df.describe()
    #         Age      Height
	#count   5.000000    5.000000
	#mean   25.800000  166.200000
	#std     9.679876   15.482248
	#min    14.000000  144.000000
	#25%    19.000000  157.000000
	#50%    25.000000  171.000000
	#75%    35.000000  178.000000
	#max    36.000000  181.000000 
	```
2) `loc` & `iloc`
	Both these methods are an absolute must-knows. They are used to retrieve subsets of the data. 
	loc :- label based indexing
	iloc:- integer based indexing

	 An example would clarify the difference between them well
 
 ```python
	df=pd.DataFrame({'Age':[14,35,25,36,19],'Height':[144,157,178,181,171]})
	
	# Let me write the code to extract the first column using both the methods 
	df.iloc[:,0]  
	df.loc[:,'Age'] 
```
3) `groupby`
	Its the go-to method for any data aggregation task, if you want to understand the inter-relationship between features. 
	```python
	#Lets up the ante by making the dataframe abit more diverse
	df=pd.DataFrame({'Age':[14,35,25,36,19],'Height':[144,157,178,181,171],'Gender':['Male','Female','Female','Male','Female']})
	
	# Suppose we are interested in the average height and age for both males and female, fret not my friend, Here comes groupby to the rescue. 
	df.groupby(by='Gender')['Height'].mean() # average height of both the classes
	# similarly 
	df.groupby(by='Gender')['Age'].mean() # average age of both the classes
```

4) `value_counts`
		If you are working with a categorical feature and would want to find the number of occurrences of each class, then `value_counts` is the method you should be using.
		Let me run you through a quick example,
	```python
	df=pd.DataFrame({'Age':[14,35,25,36,19],'Height':[144,157,178,181,171],'Gender':['Male','Female','Female','Male','Female']})
	
	#to find the number males and females in the data, we could use 
	df['Gender'].value_counts()
```

5) `pivot_table`
	- It's the same pivot table which used to give us nightmares in Excel. But Pandas makes it a tad bit easy is what i believe. 
	- With pivot table, you can build super complex aggregations, sometimes the code just bends my mind. 
	- Here is the syntax for the curious few of you: 
	  `pandas.pivot_table(_data_, _values=None_, _index=None_, _columns=None_, _aggfunc='mean'_, _fill_value=None_, _margins=False_, _dropna=True_, _margins_name='All'_, _observed=<no_default>_, _sort=True_)`

	
	```python
	df=pd.DataFrame({'Age':[14,35,25,36,19],'Height':[144,157,178,181,171],'Gender':['Male','Female','Female','Male','Female']})
	
	pd.pivot_table(df,values=['Age','Height'],index='Gender',aggfunc='mean')
	#Output
	#Gender
	#Female  26.333333  168.666667
	#Male    25.000000  162.500000
	``` 
6) `apply`
	-  If you wanna apply a function to all the entries of a column, then apply will make your life simple. 
	- Just define the operation as a regular function and then just *apply* it.
	- If you want to go more pythonic, then try lambda functions(they just make the code look more sleek)
	```python
	def lower(s: str)-> str:
		return s.lower()
	
	df=pd.DataFrame({'Age':[14,35,25,36,19],'Height':[144,157,178,181,171],'Gender':['Male','Female','Female','Male','Female']})
	
	df['Gender']=df['Gender'].apply(lower)
	```
7) `replace`
	- The name of the method simply explains what it does, replaces the values you want to be replaced with the given replacement. 
	```python
	df=pd.DataFrame({'Age':[24,35,25,36,29],'Height':[144,157,178,181,171],'Gender':['Male','Female','Female','Male','Female'],'Marital Status':['Not Interested','Alone','Married','Depressed','Single']})
	
	df.replace(to_replace=['Not Interested','Alone','Depressed'],value='Single')
	```

Pandas is a great library and has plenty more mind boggling methods which can make our lives simpler. 

Head over to the official [Pandas](https://pandas.pydata.org/docs/index.html)documentation to read more about them. 

Happy Reading! 



