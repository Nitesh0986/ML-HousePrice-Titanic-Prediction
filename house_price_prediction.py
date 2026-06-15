import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Sample House Price Dataset
data = {
    'Area': [1000,1200,1500,1800,2000,2200,2500,2800],
    'Price': [20,24,30,36,40,44,50,56]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Area']]
y = df['Price']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# R2 Score
print("R2 Score:", r2_score(y_test, predictions))

# Plot
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()