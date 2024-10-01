import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'Size': [1500, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'Price': [300000, 320000, 340000, 360000, 200000, 315000, 450000, 470000, 280000, 330000]
}
df = pd.DataFrame(data)

plt.scatter(df['Size'], df['Price'], color='blue')
plt.xlabel('Size of House (sq ft)')
plt.ylabel('Price of House ($)')
plt.title('House Price vs Size')
plt.show()

X = df[['Size']] 
y = df['Price']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

plt.scatter(X, y, color='blue')  
plt.plot(X, model.predict(X), color='red')  
plt.xlabel('Size of House (sq ft)')
plt.ylabel('Price of House ($)')
plt.title('Linear Regression Model')
plt.show()

new_house_size = [[1800]]  
predicted_price = model.predict(new_house_size)
print(f"Predicted price for a house of size 1800 sq ft: ${predicted_price[0]:.2f}")
