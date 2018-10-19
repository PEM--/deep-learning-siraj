import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# Read data
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

# Train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

# Infere prediction on read data
predictions = body_reg.predict(x_values)

# Visualize
plt.scatter(x_values, y_values)
plt.plot(x_values, predictions)
plt.title('Linear regression for predicting animal weight based on their brain weight')
plt.xlabel('Brain weight')
plt.ylabel('Body weight')
plt.show()