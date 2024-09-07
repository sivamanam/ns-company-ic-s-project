from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the saved model
modelnb = joblib.load('naive_bayes_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the file is in the request
        if 'file' not in request.files:
            return render_template('index.html', accuracy="No file uploaded")
        
        file = request.files['file']
        
        # Check if the file is a CSV
        if file.filename == '':
            return render_template('index.html', accuracy="No selected file")
        
        if file and file.filename.endswith('.csv'):
            # Read the uploaded CSV file
            data = pd.read_csv(file)
            
            # Prepare the data
            le_stability = LabelEncoder()
            data['Thermal Stability (Pass/Fail)'] = le_stability.fit_transform(data['Thermal Stability (Pass/Fail)'])
            
            x_new = data.iloc[:, :-1].values  # Features (all columns except the last)
            y_new = data.iloc[:, -1].values   # Target (the last column)

            # Predict the target using the loaded model
            predictions_encoded = modelnb.predict(x_new)

            # Calculate the accuracy
            ac = accuracy_score(predictions_encoded, y_new)
            accuracy_result = f"Model Accuracy: {ac * 100:.2f}%"
            
            # Render the same page with the accuracy result
            return render_template('index.html', accuracy=accuracy_result)
    
    # For GET request, just render the page without any result
    return render_template('index.html', accuracy=None)

if __name__ == '__main__':
    app.run(debug=True, port=5588)
