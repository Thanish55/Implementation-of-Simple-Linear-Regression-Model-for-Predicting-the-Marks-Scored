# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 

2.Set variables for assigning dataset values. 

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph. 

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: THANISH N
RegisterNumber: 212223220117
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

## DATASET:
![image](https://github.com/Thanish55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151115339/73bd2add-2a91-4e05-ba9a-df264ead4c8f)
## HEAD VALUES:
![image](https://github.com/Thanish55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151115339/0f8d00a5-ec11-4191-9bbb-7dac3eaa8791)
## TAIL VALUES:
![image](https://github.com/Thanish55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151115339/561b3dc9-1c4b-418f-a0b3-9466fb1fc9e5)
## X AND Y VALUES:
![image](https://github.com/Thanish55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151115339/bb60a1d0-b4d4-46b4-b5d9-bd8034c17e10)
## PREDICTION VALUE OF X AND Y:
![image](https://github.com/Thanish55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151115339/5237f81b-94c7-4c57-96cb-5b613f1e0cc1)
## MSE,MAE and RMSE:
![image](https://github.com/Thanish55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151115339/9e8591e8-fa77-4026-9e6b-afb64a1b2cf9)
## Training Set:
![image](https://github.com/Thanish55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151115339/f94bbcd9-c2b4-48ef-b332-c12d0f69cdd0)
## Testing Set:
![image](https://github.com/Thanish55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151115339/1dd58314-bb81-4c6a-bab5-2f8c3e80423f)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
