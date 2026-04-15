import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load clean dataset
df = pd.read_csv("../dataset/cleaned_expenses.csv")

# Convert date
df['Date'] = pd.to_datetime(df['Date'])

# Extract month
df['Month'] = df['Date'].dt.to_period('M')

# Monthly expense
monthly = df.groupby('Month')['Amount'].sum().sort_index()

# Convert to model format
X = np.arange(len(monthly)).reshape(-1, 1)
y = monthly.values

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next 3 months
future_months = np.arange(len(monthly), len(monthly)+3).reshape(-1, 1)
predictions = model.predict(future_months)

print("📊 Monthly Data:\n", monthly)
print("\n🔮 Next 3 Months Prediction:")

for i, val in enumerate(predictions, 1):
    print(f"Month +{i}: ₹{round(val,2)}")
    
pred_df = pd.DataFrame({
    "Future_Month": ["Month+1", "Month+2", "Month+3"],
    "Predicted_Expense": predictions 
})
pred_df.to_csv("../dataset/prediction.csv", index = False)
print("\n Prediction File Saved")
