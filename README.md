# Developer Name : Manoj M
# Reg no : 212221240027
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 
### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Choose FinalGrade as the target variable for forecasting.
3. Divide FinalGrade into training (80%) and testing (20%) sets.
4. Fit a SARIMA model with specified order and seasonal parameters on the training set.
5. Generate and plot forecasted values against actual data, then print forecasted grades.
### PROGRAM:
```# Import necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Sample Data - replace with actual data loading process if using a file
df = pd.read_csv('/content/student_performance(1).csv')

# Encode categorical features (like Gender and ParentalSupport)
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_parental_support = LabelEncoder()
df['ParentalSupport'] = le_parental_support.fit_transform(df['ParentalSupport'])

# Define the series we want to forecast (e.g., FinalGrade)
series = df['FinalGrade']

# Split data into training and test sets
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# Fit the SARIMA model
# SARIMA model parameters: order=(p,d,q), seasonal_order=(P,D,Q,s)
# Here we are using seasonal_order=(1, 1, 1, s), where s=4 (assuming hypothetical seasonality every 4 entries)
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
model_fit = model.fit(disp=False)

# Forecast
forecast = model_fit.forecast(steps=len(test))
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(series.index, series, label='Actual Final Grade')
plt.plot(test.index, forecast, color='red', label='Forecasted Final Grade')
plt.title('SARIMA Forecast for Final Grade')
plt.xlabel('Index')
plt.ylabel('Final Grade')
plt.legend()
plt.show()
# Print the forecasted values
print("Forecasted Final Grade values:", forecast)
```

### OUTPUT:
![Untitled](https://github.com/user-attachments/assets/eaac590f-4bd3-485c-85d1-a31005729504)


### RESULT:
Thus the program run successfully based on the SARIMA model.
