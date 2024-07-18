from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved models
try:
    model_heart = pickle.load(open('heart.pkl', 'rb'))
    model_diabetes = pickle.load(open('diabetes_model.pkl', 'rb'))
    model_breast = pickle.load(open('breast.pkl', 'rb'))
    model_perkinson = pickle.load(open('parkinsons_model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading models: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/heart')
def heart():
    return render_template('heart.html')
@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/breast')
def breast():
    return render_template('breast.html')
@app.route('/perkinson')
def perkinson():
    return render_template('perkinson.html')


@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    try:
        print("Form data received for heart prediction:", request.form)
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model_heart.predict(input_data)

        if prediction[0] == 1:
            result = 'The Person has Heart Disease'
        else:
            result = 'The Person does not have Heart Disease'

        return render_template('heart.html', result=result)
    except Exception as e:
        print(f"Error occurred in heart disease prediction: {str(e)}")
        return render_template('heart.html', result=f"Error: {str(e)}")

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        print("Form data received for diabetes prediction:", request.form)
        pregnancy = float(request.form['pregnancy'])
        glucose = float(request.form['glucose'])
        bloodpressure = float(request.form['bloodpressure'])
        skinthickness = float(request.form['skinthickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes = float(request.form['diabetes'])
        age = float(request.form['age'])

        input_data = np.array([[pregnancy, glucose, bloodpressure, skinthickness, insulin, bmi, diabetes, age]])
        prediction = model_diabetes.predict(input_data)

        if prediction[0] == 1:
            result = 'The Person has Diabetes'
        else:
            result = 'The Person does not have Diabetes'

        return render_template('diabetes.html', result=result)
    except Exception as e:
        print(f"Error occurred in diabetes prediction: {str(e)}")
        return render_template('diabetes.html', result=f"Error: {str(e)}")

@app.route('/predict_breast', methods=['POST'])
def predict_breast():
    try:
        print("Form data received for breast cancer prediction:", request.form)
        radius_mean = float(request.form['radius_mean'])
        texture_mean = float(request.form['texture_mean'])
        parameter_mean = float(request.form['parameter_mean'])
        area_mean = float(request.form['area_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        compactness_mean = float(request.form['compactness_mean'])
        concavity_mean = float(request.form['concavity_mean'])
        concavepoints_mean = float(request.form['concavepoints_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
        radius_se = float(request.form['radius_se'])
        texture_se = float(request.form['texture_se'])
        perimeter_se = float(request.form['perimeter_se'])
        area_se = float(request.form['area_se'])
        smoothness_se = float(request.form['smoothness_se'])
        compactness_se = float(request.form['compactness_se'])
        concavity_se = float(request.form['concavity_se'])
        concavepoints_se = float(request.form['concavepoints_se'])
        symmetry_se = float(request.form['symmetry_se'])
        fractal_dimension_se = float(request.form['fractal_dimension_se'])
        radius_worst = float(request.form['radius_worst'])
        texture_worst = float(request.form['texture_worst'])
        perimeter_worst = float(request.form['perimeter_worst'])
        area_worst = float(request.form['area_worst'])
        smoothness_worst = float(request.form['smoothness_worst'])
        compactness_worst=float(request.form['compactness_worst'])
        concavity_worst = float(request.form['concavity_worst'])
        concavepoints_worst = float(request.form['concavepoints_worst'])
        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst = float(request.form['fractal_dimension_worst'])

        input_data = np.array([[radius_mean, texture_mean, parameter_mean, area_mean, smoothness_mean, 
                                compactness_mean, concavity_mean, concavepoints_mean, symmetry_mean, 
                                fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, 
                                smoothness_se, compactness_se, concavity_se, concavepoints_se, symmetry_se, 
                                fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, 
                                smoothness_worst,compactness_worst, concavity_worst, concavepoints_worst, symmetry_worst, 
                                fractal_dimension_worst]])

        prediction = model_breast.predict(input_data)

        if prediction[0] == 1:
            result = 'The Breast cancer is Benign'
        else:
            result = 'The Breast cancer is Malignant'

        return render_template('breast.html', result=result)
    except Exception as e:
        print(f"Error occurred in breast cancer prediction: {str(e)}")
        return render_template('breast.html', result=f"Error: {str(e)}")

@app.route('/predict_perkenson', methods=['POST'])
def predict_perkenson():
    try:
        print("Form data received for Parkinson's prediction:", request.form)

        # Extract form data
        age = float(request.form['age'])
        gender = float(request.form['gender'])
        ethnicity = float(request.form['ethnicity'])
        education = float(request.form['education'])
        bmi = float(request.form['bmi'])
        smoking = float(request.form['smoking'])
        alcoholconsumption = float(request.form['alcoholconsumption'])
        dailyactivity = float(request.form['dailyactivity'])
        dietquality = float(request.form['dietquality'])
        sleepquality = float(request.form['sleepquality'])
        familyrelations = float(request.form['familyrelations'])
        traumatic = float(request.form['traumatic'])
        hypertension = float(request.form['hypertension'])
        diabetes = float(request.form['diabetes'])
        depression = float(request.form['depression'])
        stroke = float(request.form['stroke'])
        systolic = float(request.form['systolic'])
        diastolicbp = float(request.form['diastolicbp'])
        cholestroltotal = float(request.form['cholestroltotal'])
        cholestrolldl = float(request.form['cholestrolldl'])
        cholestrolhdl = float(request.form['cholestrolhdl'])
        cholestroltrigly = float(request.form['cholestroltrigly'])
        updrs = float(request.form['updrs'])
        Moca = float(request.form['Moca'])
        Functional = float(request.form['Functional'])
        tremor = float(request.form['tremor'])
        rigidity = float(request.form['rigidity'])
        Bradykinesia = float(request.form['Bradykinesia'])
        Posturalinstability = float(request.form['Posturalinstability'])
        SpeechProblems = float(request.form['SpeechProblems'])
        SleepDisorder = float(request.form['SleepDisorder'])
        constipation = float(request.form['constipation'])

        # Prepare the input data
        input_data = np.array([[age, gender, ethnicity, education, bmi, smoking, alcoholconsumption, dailyactivity, dietquality, sleepquality, familyrelations, traumatic, hypertension, diabetes, depression, stroke, systolic, diastolicbp, cholestroltotal, cholestrolldl, cholestrolhdl, cholestroltrigly, updrs, Moca, Functional, tremor, rigidity, Bradykinesia, Posturalinstability, SpeechProblems, SleepDisorder, constipation]])

        # Make a prediction using the loaded model
        prediction = model_perkinson.predict(input_data)

        # Interpret the prediction
        if prediction[0] == 1:
            result = 'Person has Parkinson'
        else:
            result = 'Person does not have Parkinson'

        return render_template('perkinson.html', result=result)
    except Exception as e:
        print(f"Error occurred in Parkinson's prediction: {str(e)}")
        return render_template('perkinson.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)


