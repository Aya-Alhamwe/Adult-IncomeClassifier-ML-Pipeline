from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd

# Load the models
svm_model = joblib.load("svm_income_model.pkl")
column_transformer = joblib.load("column_transformed.pkl")

# Assuming you have a dataframe or lists for the options:
# Example: these could come from your training dataset or predefined categories
workclass_options = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
marital_status_options = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed']
occupation_options = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farmers-fishers', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
relationship_options = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
race_options = ['White','Black','Others']
sex_options = ['Male', 'Female']
native_country_options = ['United-States','Others']
native_country_options = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'Nicaragua', 'Scotland', 'Mexico', 'Ireland', 'Hong', 'Holand-Netherlands']
education_options = ['College', 'Bachelors', 'Middle-School', 'Masters', 'Doctorate', 'Primary', 'Preschool']

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def form_page():
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Income Prediction</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f0f2f5; }}
            form {{ max-width: 500px; margin: auto; padding: 20px; background-color: #fff; border-radius: 10px; box-shadow: 0 0 15px rgba(0, 0, 0, 0.2); }}
            .form-group {{ margin-bottom: 15px; text-align: left; }}
            label {{ display: block; margin-bottom: 5px; font-weight: bold; }}
            input, select {{ width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 5px; box-sizing: border-box; }}
            button {{ width: 100%; padding: 12px; background-color: #007BFF; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }}
            button:hover {{ background-color: #0056b3; }}
            .result {{ margin-top: 20px; text-align: center; font-size: 20px; color: #333; }}
            h2 {{ text-align: center; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h2>Income Prediction</h2>
        <form action="/predict" method="post" onsubmit="event.preventDefault(); predictIncome();">
            <div class="form-group">
                <label for="workclass">Workclass</label>
                <select name="workclass" id="workclass" required>
                    {''.join([f'<option value="{option}">{option}</option>' for option in workclass_options])}
                </select>
            </div>
            <div class="form-group">
                <label for="marital_status">Marital Status</label>
                <select name="marital_status" id="marital_status" required>
                    {''.join([f'<option value="{option}">{option}</option>' for option in marital_status_options])}
                </select>
            </div>
            <div class="form-group">
                <label for="occupation">Occupation</label>
                <select name="occupation" id="occupation" required>
                    {''.join([f'<option value="{option}">{option}</option>' for option in occupation_options])}
                </select>
            </div>
            <div class="form-group">
                <label for="relationship">Relationship</label>
                <select name="relationship" id="relationship" required>
                    {''.join([f'<option value="{option}">{option}</option>' for option in relationship_options])}
                </select>
            </div>
            <div class="form-group">
                <label for="race">Race</label>
                <select name="race" id="race" required>
                    {''.join([f'<option value="{option}">{option}</option>' for option in race_options])}
                </select>
            </div>
            <div class="form-group">
                <label for="sex">Sex</label>
                <select name="sex" id="sex" required>
                    {''.join([f'<option value="{option}">{option}</option>' for option in sex_options])}
                </select>
            </div>
            <div class="form-group">
                <label for="native_country">Native Country</label>
                <select name="native_country" id="native_country" required>
                    {''.join([f'<option value="{option}">{option}</option>' for option in native_country_options])}
                </select>
            </div>
            <div class="form-group">
                <label for="education">Education</label>
                <select name="education" id="education" required>
                    {''.join([f'<option value="{option}">{option}</option>' for option in education_options])}
                </select>
            </div>
            <div class="form-group">
                <label for="capital_diff">Capital Diff (Low/High)</label>
                <input type="text" name="capital_diff" id="capital_diff" required>
            </div>
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" name="age" id="age" required>
            </div>
            <div class="form-group">
                <label for="hours_per_week">Hours per Week</label>
                <input type="number" name="hours_per_week" id="hours_per_week" required>
            </div>
            <div class="form-group">
                <label for="education_num">Education Num</label>
                <input type="number" name="education_num" id="education_num" required>
            </div>
            <button type="submit">Predict</button>
        </form>
        <div class="result" id="predictionResult"></div>
        <script>
            async function predictIncome() {{
                const formData = new FormData(document.querySelector('form'));
                const response = await fetch('/predict', {{
                    method: 'POST',
                    body: formData
                }});
                const result = await response.text();
                document.getElementById('predictionResult').innerHTML = result;
            }}
        </script>
    </body>
    </html>
    """
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, workclass: str = Form(...), marital_status: str = Form(...), occupation: str = Form(...),
                  relationship: str = Form(...), race: str = Form(...), sex: str = Form(...), native_country: str = Form(...),
                  education: str = Form(...), capital_diff: str = Form(...), age: float = Form(...),
                  hours_per_week: float = Form(...), education_num: float = Form(...)):
    try:
        # Create a DataFrame from user input
        input_data = pd.DataFrame({
            "workclass": [workclass],
            "marital.status": [marital_status],
            "occupation": [occupation],
            "relationship": [relationship],
            "race": [race],
            "sex": [sex],
            "native.country": [native_country],
            "education": [education],
            "capital_diff": [capital_diff],
            "age": [age],
            "hours.per.week": [hours_per_week],
            "education.num": [education_num]
        })

        # Transform the input data
        input_transformed = column_transformer.transform(input_data)

        # Predict
        prediction = svm_model.predict(input_transformed)[0]
        result = "Income > 50K" if prediction == 1 else "Income <= 50K"

        return f"<div class='result' style='color: green; font-weight: bold; margin-top: 20px;'>Prediction: {result}</div>"

    except Exception as e:
        return f"<div class='result' style='color: red;'>An error occurred: {str(e)}</div>"
