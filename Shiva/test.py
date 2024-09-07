import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the saved model
modelnb = joblib.load('naive_bayes_model.pkl')

# Ask for the new dataset file path
file_path = input("Please enter the CSV file path for prediction: ")

# Load the new dataset
new_data = pd.read_csv(file_path)

# Encode 'Thermal Stability (Pass/Fail)' for the new data if it's categorical
le_stability = LabelEncoder()
new_data['Thermal Stability (Pass/Fail)'] = le_stability.fit_transform(new_data['Thermal Stability (Pass/Fail)'])

# Split the data into features and target
x_new = new_data.iloc[:, :-1].values  # Features (all columns except the last)
y_new = new_data.iloc[:, -1].values   # Target (the last column, e.g., 'Thermal Stability')

# Predict the target using the loaded model
predictions_encoded = modelnb.predict(x_new)

# Calculate and print the accuracy score
ac = accuracy_score(predictions_encoded, y_new)
print(f"Model Accuracy: {ac * 100:.2f}%")

# Optionally, save the results
output_file = "predicted_results.csv"
new_data['Predicted Stability'] = predictions_encoded
new_data.to_csv(output_file, index=False)
print(f"Predicted results saved to {output_file}")

