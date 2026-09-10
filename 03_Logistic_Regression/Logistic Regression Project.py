import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

ad_data = pd.read_csv('advertising.csv')

print('___'*500)
print(ad_data.head())
print('___'*500)
print(ad_data.info())
print('___'*500)


#===============================================================================================================
# TRAINING MODEL
#===============================================================================================================

from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report


print('=============================================== ACCURACY TEST =============================================================')
print(classification_report(y_test,predictions))



# ===============================================================================================================
# INTERACTIVE PREDICTION: WILL A NEW USER CLICK THE AD?
# ===============================================================================================================
print('\n' + '==='*30)
print("🔮 Let's Predict a New User's Behavior!")
print('==='*30)

# Step 1: Collect data from the user via the console
time_spent = float(input("Enter Daily Time Spent on Site in minutes (e.g., 65.5): "))
age = int(input("Enter User's Age (e.g., 30): "))
income = float(input("Enter Area Income (e.g., 60000.00): "))
internet_usage = float(input("Enter Daily Internet Usage in minutes (e.g., 200.5): "))
is_male = int(input("Is the user Male? (Type 1 for Yes, 0 for No): "))

# Step 2: Store the collected data in a pandas DataFrame
# Why a DataFrame? So the column names match exactly what the model studied during training!
new_user_data = pd.DataFrame({
    'Daily Time Spent on Site': [time_spent],
    'Age': [age],
    'Area Income': [income],
    'Daily Internet Usage': [internet_usage],
    'Male': [is_male]
})

# Step 3: Feed the new data to our trained 'logmodel' for a prediction
new_prediction = logmodel.predict(new_user_data)

# Step 4: Display the result cleanly
print('\n' + '==='*30)
if new_prediction[0] == 1:
    print("🎯 PREDICTION: This user WILL likely CLICK on the ad! (Output: 1)")
else:
    print("🛑 PREDICTION: This user will likely IGNORE the ad. (Output: 0)")
print('==='*30)