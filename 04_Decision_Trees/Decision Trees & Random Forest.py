import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#Read the CSV file
loans = pd.read_csv('loan_data.csv')

#Check the data set
print('___'*500)
print(loans.head())
print('___'*500)
print(loans.info())
print('___'*500)
print(loans.describe())
print('___'*500)
print()


plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red', bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.ylabel('Frequency')
plt.show()

sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
plt.show()

#=================================================================================================
# LET'S START TO TRAIN OUR MODEL
#=================================================================================================

cat_feats = ['purpose']

final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True, dtype=int)

print(final_data.info())

#==================================================================================================
# Train Test Split
#==================================================================================================

from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print('___'*500)
print(classification_report(y_test,predictions))
print('___'*500)
print(confusion_matrix(y_test,predictions))
print('___'*500)


#==================================================================================================
# Training the Random Forest model
#==================================================================================================

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=600, random_state=101)
rfc.fit(X_train, y_train)

rfc_predictions = rfc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print('___'*100)
print("RANDOM FOREST CLASSIFICATION REPORT")
print('___'*100)
print(classification_report(y_test, rfc_predictions))

print('___'*100)
print("RANDOM FOREST CONFUSION MATRIX")
print(confusion_matrix(y_test, rfc_predictions))
print('___'*100)