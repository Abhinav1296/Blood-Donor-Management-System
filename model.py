import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
from datetime import datetime, timedelta

# Load and preprocess the dataset
def load_and_preprocess_data():
    data = pd.read_csv('Blood_Donor_Data.csv')
    
    # Encode categorical variables
    le_blood_type = LabelEncoder()
    le_city = LabelEncoder()
    le_blood_bank = LabelEncoder()
    
    data['Blood_Type'] = le_blood_type.fit_transform(data['Blood_Type'])
    data['City'] = le_city.fit_transform(data['City'])
    data['Blood_Bank'] = le_blood_bank.fit_transform(data['Blood_Bank'])
    
    # Simulate Blood_Needed_Date (add random future days for training)
    np.random.seed(42)
    current_date = datetime.now()
    data['Blood_Needed_Date'] = [
        (current_date + timedelta(days=np.random.randint(1, 31))).strftime('%Y-%m-%d')
        for _ in range(len(data))
    ]
    
    # Convert Blood_Needed_Date to days from current date
    data['Blood_Needed_Days'] = [
        (datetime.strptime(date, '%Y-%m-%d') - current_date).days
        for date in data['Blood_Needed_Date']
    ]
    
    # Rename Units_Available to Units_Needed for training
    data['Units_Needed'] = data['Units_Available']
    
    # Introduce controlled noise
    data['Available'] = data['Available'].apply(lambda x: 1 - x if np.random.random() < 0.05 else x)
    
    # Features and target
    X = data[['Age', 'Blood_Type', 'City', 'Blood_Bank', 'Units_Needed', 'Blood_Needed_Days']]
    y = data['Available']
    
    return X, y, {
        'blood_type': le_blood_type,
        'city': le_city,
        'blood_bank': le_blood_bank
    }

# Train the model
def train_model():
    X, y, encoders = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Save the model and encoders
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    with open('feature_count.pkl', 'wb') as f:
        pickle.dump(X.shape[1], f)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Trained Model Accuracy: {accuracy:.2f}%")
    
    return model, encoders

# Load or train model
def get_model():
    expected_features = 6
    retrain = False
    
    if os.path.exists('feature_count.pkl'):
        with open('feature_count.pkl', 'rb') as f:
            saved_features = pickle.load(f)
        if saved_features != expected_features:
            print(f"Feature mismatch: Expected {expected_features}, got {saved_features}. Retraining...")
            retrain = True
    else:
        retrain = True
    
    if not os.path.exists('model.pkl') or not os.path.exists('encoders.pkl') or retrain:
        model, encoders = train_model()
    else:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
    
    return model, encoders

# Predict availability
def predict_availability(inputs):
    model, encoders = get_model()
    
    try:
        blood_type = encoders['blood_type'].transform([inputs['Blood_Type']])[0]
        city = encoders['city'].transform([inputs['City']])[0]
        blood_bank = encoders['blood_bank'].transform([inputs['Blood_Bank']])[0]
    except ValueError as e:
        return None, f"Invalid input: {str(e)}"
    
    try:
        needed_date = datetime.strptime(inputs['Blood_Needed_Date'], '%Y-%m-%d')
        current_date = datetime.now()
        blood_needed_days = (needed_date - current_date).days
    except ValueError:
        return None, "Invalid date format"
    
    # Load the full dataset
    data = pd.read_csv('Blood_Donor_Data.csv')
    
    # Hierarchical filtering
    # Step 1: Filter by Blood_Type
    filtered_data = data[data['Blood_Type'] == inputs['Blood_Type']]
    if filtered_data.empty:
        return {'available': 'No', 'probability': 0.0}, None  # No data for this blood type
    
    # Step 2: Filter by City
    filtered_data = filtered_data[filtered_data['City'] == inputs['City']]
    if filtered_data.empty:
        return {'available': 'No', 'probability': 0.0}, None  # No data for this city
    
    # Step 3: Filter by Blood_Bank
    filtered_data = filtered_data[filtered_data['Blood_Bank'] == inputs['Blood_Bank']]
    if filtered_data.empty:
        return {'available': 'No', 'probability': 0.0}, None  # No data for this blood bank
    
    # Step 4: Calculate available units
    available_units = filtered_data['Units_Available'].sum()
    
    features = np.array([[
        inputs['Age'],
        blood_type,
        city,
        blood_bank,
        int(inputs['Units_Needed']),
        blood_needed_days
    ]])
    
    try:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][prediction]
        # Override prediction based on Units_Available vs Units_Needed
        if available_units >= int(inputs['Units_Needed']):
            final_prediction = 1  # Yes
        else:
            final_prediction = 0  # No
    except ValueError as e:
        return None, f"Model error: {str(e)}. Please delete 'model.pkl', 'encoders.pkl', and 'feature_count.pkl' and restart."
    
    return {
        'available': 'Yes' if final_prediction == 1 else 'No',
        'probability': probability * 100  # Keep model's confidence unchanged
    }, None