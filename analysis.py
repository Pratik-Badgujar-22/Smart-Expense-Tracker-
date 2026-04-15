import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../dataset/expenses.csv")

# 🔥 FORCE CLEAN DATE (important)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# Remove invalid dates (if any)
df = df.dropna(subset=['Date'])

# 🔥 SORT properly
df = df.sort_values(by='Date')

# Extract Month
df['Month'] = df['Date'].dt.to_period('M')

# -------------------------------
# Analysis

monthly = df.groupby('Month')['Amount'].sum().sort_index()
print("Monthly Expense:\n", monthly)

category = df.groupby('Category')['Amount'].sum()
print("\nCategory Expense:\n", category)

# -------------------------------
# Save clean dataset
df.to_csv("../dataset/cleaned_expenses.csv", index=False)

print("\n✅ Cleaned dataset saved")