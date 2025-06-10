from flask import Flask, render_template, request
import pandas as pd
from model import predict_availability

app = Flask(__name__)

# Load dataset for display
dataset = pd.read_csv('Blood_Donor_Data.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            inputs = {
                'Age': int(request.form['age']),
                'Blood_Type': request.form['blood_type'],
                'City': request.form['city'],
                'Blood_Bank': request.form['blood_bank'],
                'Units_Needed': int(request.form['units_needed']),
                'Blood_Needed_Date': request.form['blood_needed_date']
            }
        except ValueError:
            return render_template('predict.html', error="Invalid input values")
        
        result, error = predict_availability(inputs)
        if error:
            return render_template('predict.html', error=error)
        return render_template('predict.html', result=result)
    
    return render_template('predict.html')

@app.route('/dataset')
def dataset_view():
    data_html = dataset.head(100).to_html(classes='table table-striped', index=False)
    return render_template('dataset.html', data_html=data_html)

if __name__ == '__main__':
    app.run(debug=True)