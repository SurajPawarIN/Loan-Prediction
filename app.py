from flask import Flask, render_template, request
import joblib
import pandas as pd
app = Flask(__name__)

# Load the trained pipeline
loaded_pipeline = joblib.load('loan_prediction_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        user_input = {
            'Gender': request.form['gender'],
            'Married': request.form['married'],
            'Dependents': request.form['dependents'],
            'Education': request.form['education'],
            'Self_Employed': request.form['self_employed'],
            'Property_Area': request.form['property_area'],
            'ApplicantIncome': float(request.form['applicant_income']),
            'CoapplicantIncome': float(request.form['coapplicant_income']),
            'LoanAmount': float(request.form['loan_amount']),
            'Loan_Amount_Term': int(request.form['loan_amount_term']),
            'Credit_History': int(request.form['credit_history'])
        }

        # Convert the user input dictionary to a DataFrame
        user_input_df = pd.DataFrame(user_input, index=[0])

        # Use the loaded pipeline to make predictions on user input
        predicted_prob = loaded_pipeline.predict_proba(user_input_df)[:, 1]
        predicted_class = loaded_pipeline.predict(user_input_df)

        return render_template('index.html', predicted_prob=predicted_prob[0], predicted_class=predicted_class[0])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
