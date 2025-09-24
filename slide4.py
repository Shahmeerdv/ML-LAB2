import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

df = pd.read_csv('Admission_Predict.csv')
columns = df.columns

df.drop("Serial No.", axis=1, inplace=True)
y = df['Chance of Admit ']
df.drop('Chance of Admit ', axis=1, inplace=True)
print(df.head())



# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# Create and train model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predict on test set
pred = lr.predict(x_test)
print(pred[:10])  # show first 10 predictions



rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
print(rmse)
