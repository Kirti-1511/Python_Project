#Importing Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

%matplotlib inline

#Reading our input data for house price prediction
import pandas as pd 
from google.colab import files

HouseDF = pd.read_csv('/content/sample_data/USA_Housing.csv')
HouseDF.head() 

HouseDF.info()
HouseDF.describe()

#Analyzing information from our data
HouseDF.info()

#plots to visualize data of House Price Prediction
sns.pairplot(HouseDF)

sns.distplot(HouseDF['Price'])
sns.heatmap(HouseDF.corr(), annot=True)

#Get Data Ready For Training a Linear Regression Model

X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]

y = HouseDF['Price']

#Split Data into Train, Test 
#X_train and y_train contain data for the training model. X_test and y_test contain data for the testing model. X and y are features and target variable names.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) 

#Creating and Training the LinearRegression Model
from sklearn.linear_model import LinearRegression 

lm = LinearRegression() 

lm.fit(X_train,y_train) 

#LinearRegression Model Evaluation
print(lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient']) 
coeff_df

#Predictions from our Linear Regression Model
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)

sns.distplot((y_test-predictions),bins=50);

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions)) 
print('MSE:', metrics.mean_squared_error(y_test, predictions)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
