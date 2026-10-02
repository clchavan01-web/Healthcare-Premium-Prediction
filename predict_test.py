import joblib
import pandas as pd
model = joblib.load("healthcare_model.pkl")
def get_prediction(age,BMI,smoker_text):
    smoker=1 if smoker_text.lower()=="yes" else 0
    input_data=pd.DataFrame([[age,BMI,smoker]],columns=["age","BMI","smoker"])
    prediction=model.predict(input_data)
    return prediction
result= get_prediction(age=30,BMI=25.0,smoker_text="no")
print(f"step 2 successfull : The loaded model predicted a premium of rs{result:.2f}")

