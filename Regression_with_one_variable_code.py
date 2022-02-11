# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

#read data
path = 'D:\\ML\Regression_with_one_variable\\ML_Regression_with_one_variable\\ex1data1.txt';
data = pd.read_csv(path,header=None,names=['Population','Profit'])

#show data details
# =============================================================================
# print('data = \n' , data.head(10))
# print('****************************')
# print('data.describe = \n' , data.describe())
# =============================================================================

#draw Data
data.plot(kind='scatter', x='Population', y='Profit' , figsize=(5,5))
                                                                      
#adding a new column called ones before the data
data.insert(0,'Ones',1) 
# =============================================================================
# #data after adding the new column
# print('data = \n' , data.head(10))
# print('****************************')  
# =============================================================================

#separate X: trainning data from y : target variable
#data.shape[0]  return number of lignes
#data.shape[1]  return number of columns

cols = data.shape[1]
X = data.iloc[: , 0 : cols-1] 
y = data.iloc[: , cols-1 : cols]

# =============================================================================
# print('******************************************************')
# print('X data = \n',X.head(10))
# print('X data = \n',y.head(10))
# =============================================================================
                                      

#Convert from data frames to numpy matrix
X= np.matrix(X.values)
Y= np.matrix(y.values)
thetha = np.matrix(np.array([0,0]))


# print("matrice=",X*X)
# print('******************************************************')
# print('X = \n',X)
# print('X.shape = \n',X.shape)
# print('thetha= \n',thetha)
# print('thetha.shape= \n',thetha.shape)
# print('y= \n',y)
# print('y.shape= \n',y.shape)
# print('******************************************************')
                                 
#Cost Function
def computeCost(X,y,thetha):
    z = np.power(((X*thetha.T)-y),2) 
    return (np.sum(z) / (2* len(X)) ) 

print('computeCost(X,y,thetha)=',computeCost(X,y,thetha)) 
print('*******************************************************')  

#GD Function:
def gradientDescent(X,y,thetha,alpha,iters):
    
    temp= np.matrix(np.zeros(thetha.shape))
    parameters = int(thetha.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X*thetha.T)-y;
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = thetha[0,j] - ((alpha / len(X)) * np.sum(term))
            
        thetha =temp;
        cost[i] = computeCost(X,y,thetha)
    
    return thetha,cost    
    

#initialize variables for learning rate and iterations:
iters = 1000
alpha = 0.01

#perform gradient descent to "fit" the model parameters:
    
g,cost = gradientDescent(X,y,thetha,alpha,iters)

print('*******************************************************') 

print(' g = \n',g) 
print(' cost = \n',cost) 
print('computeCost(X,y,thetha)=' , computeCost(X, y, g)) 
  
print('*******************************************************') 

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0,0] + g[0,1]*x
#print('f=\n',f)  

#draw the line
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit  vs. Population Profit')  

#draw the cost function
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost , 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs Training Epoch')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      