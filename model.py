import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load customer dataset



# Replace backslashes with forward slashes

data = pd.read_csv('Customer.csv')

# Display the first few rows to verify
print(data.head())

# Step 1: Check for missing values in the dataset
missing_values = data.isnull().sum()




# Select the renamed features
selected_features = data[['Total_Spend',
                          'Num_of_Purchases',
                          'Last_Purchase_Days_Ago']]
#


# Display missing values and the selected features

print(missing_values)
print(selected_features.head())

# Check the data types of the columns
print(data.dtypes)

# Convert 'target_churn' column to integer
# First, handle any non-numeric values by coercing to NaN, then fill or handle NaNs appropriately
data['Target_Churn'] = pd.to_numeric(data['Target_Churn'], errors='coerce').fillna(0).astype(int)

# Verify the conversion
print(data['Target_Churn'].head())
print(data.dtypes)


# Preprocess the data
X = data[['Total_Spend',
          'Num_of_Purchases',
          'Last_Purchase_Days_Ago']]
y = data['Target_Churn']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model1 = LogisticRegression()
model1.fit(X_train, y_train)

# Predict on the test set and display results (optional)
predictions = model1.predict(X_test)
print(predictions)


# Save the trained model
joblib.dump(model1, 'churn_model.pkl')
print("Model training completed and saved.")


