import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
customers = pd.read_csv("Ecommerce Customers")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(customers.head())
print('__'*500)
print(customers.describe())
print('__'*500)
print(customers.info())

#===============================================================================================================
# TRY TO PREDICT AN ANSWER BY YOURSELVES USING BELOW GRAPHS
#===============================================================================================================
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

# More time on site, more money spent.
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
plt.show()

#After reviewing graph we can predict that more time on app more spend
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
plt.show()

sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)
plt.show()

sns.pairplot(customers)
plt.show()

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
plt.show()

#===============================================================================================================
# TRAINING MODEL
#===============================================================================================================
y = customers['Yearly Amount Spent']
x = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


predictions = regressor.predict( X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



coeffecients = pd.DataFrame(regressor.coef_,x.columns)
coeffecients.columns = ['Coeffecient']
print(coeffecients)






