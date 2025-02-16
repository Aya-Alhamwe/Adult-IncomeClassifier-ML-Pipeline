from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib

app = FastAPI()

# Load the pre-trained model and scaler
model = joblib.load(r'C:\Users\ayaaa\Downloads\task\xgb_balanced_model.pkl')
scaler = joblib.load(r'C:\Users\ayaaa\Downloads\task\scaler.pkl')

# Define the form to collect user data
@app.get("/", response_class=HTMLResponse)
async def form_page():
    html_content = """
    <html>
        <head>
            <title>Salary Prediction</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f9;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    flex-direction: column;
                }
                .container {
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    width: 300px;
                }
                h2 {
                    color: #333;
                }
                input, select {
                    width: 100%;
                    padding: 10px;
                    margin: 10px 0;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }
                button {
                    width: 100%;
                    padding: 10px;
                    background-color: #28a745;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                button:hover {
                    background-color: #218838;
                }
                .result {
                    margin-top: 20px;
                    font-size: 18px;
                    font-weight: bold;
                }
            </style>
            <script>
                async function predictSalary(event) {
                    event.preventDefault();
                    const form = document.getElementById('prediction-form');
                    const formData = new FormData(form);
                    const response = await fetch('/predict/', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    document.getElementById('result').innerHTML = "Predicted Salary: " + data.salary;
                }
            </script>
        </head>
        <body>
            <div class="container">
                <h2>Predict Your Salary</h2>
                <form id="prediction-form" onsubmit="predictSalary(event)">
                    <label for="workclass">Workclass</label>
                    <select name="workclass" id="workclass">
                        <option value="0">Private</option>
                        <option value="1">Self-emp-not-inc</option>
                        <option value="2">Self-emp-inc</option>
                        <option value="3">Federal-gov</option>
                        <option value="4">Local-gov</option>
                        <option value="5">State-gov</option>
                        <option value="6">Without-pay</option>
                        <option value="7">Never-worked</option>
                    </select>

                    <label for="education">Education</label>
                    <select name="education" id="education">
                        <option value="0">Bachelors</option>
                        <option value="1">Masters</option>
                        <option value="2">Doctorate</option>
                        <option value="3">HS-grad</option>
                        <option value="4">Some-college</option>
                    </select>

                    <label for="occupation">Occupation</label>
                    <select name="occupation" id="occupation">
                        <option value="0">Tech-support</option>
                        <option value="1">Craft-repair</option>
                        <option value="2">Other-service</option>
                        <option value="3">Sales</option>
                        <option value="4">Exec-managerial</option>
                    </select>

                    <label for="sex">Sex</label>
                    <select name="sex" id="sex">
                        <option value="0">Male</option>
                        <option value="1">Female</option>
                    </select>

                    <label for="hours-per-week">Hours per week</label>
                    <input type="number" name="hours_per_week" id="hours_per_week" required>

                    <button type="submit">Predict Salary</button>
                </form>
                <div id="result" class="result"></div>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict_salary(workclass: int = Form(...), education: int = Form(...), occupation: int = Form(...),
                         sex: int = Form(...), hours_per_week: int = Form(...)):
   
    input_data = pd.DataFrame({
        'workclass': [workclass],
        'education': [education],
        'occupation': [occupation],
        'sex': [sex],
        'hours.per.week': [hours_per_week],
    })

  
    required_features = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status',
                         'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss',
                         'hours.per.week', 'native.country']

    for col in required_features:
        if col not in input_data.columns:
            input_data[col] = 0  

  
    input_data = input_data[required_features]

    
    input_data_scaled = scaler.transform(input_data)

   
    prediction = model.predict(input_data_scaled)
   
    salary = ">50K" if prediction[0] == 1 else "<=50K"

    
    return {"salary": salary}
