from m1 import el
import pandas as pd
# import math
import joblib
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
import numpy as np
from flask import Flask, request,render_template

app = Flask(__name__)
model = joblib.load("model.joblib")

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict() 
    input_data = pd.DataFrame(data, index=[0]) 
    
    input_data["seller_type"] = input_data["seller_type"].map({'OWNER':0,'AGENT':1,'BUILDER':2})  
    input_data["bedroom"] = input_data["bedroom"].astype(int)
    input_data["layout_type"] = input_data["layout_type"].map({'BHK':0,'RK':1})
    input_data["property_type"] = input_data["property_type"].map({'Apartment':0,'Studio Apartment':1,'Independent House':2,'Independent Floor':3,'Villa':4,'Penthouse':5})
    input_data["area"] = np.log(input_data["area"].astype(float))
    input_data["furnish_type"] = input_data["furnish_type"].map({'Furnished':0,'Semi-Furnished':1,'Unfurnished':2})
    input_data["bathroom"] = input_data["bathroom"].astype(int)
    input_data["city"] = input_data["city"].map({'Ahmedabad':0,'Bangalore':1,'Chennai':2,'Delhi':3,'Hyderabad':4,'Kolkata':5,'Mumbai':6,'Pune':7})   
    encoded_localities = el.transform(input_data["localities"])


    input_data["localities"] = encoded_localities     
    predictions = model.predict(input_data)
    predictions = np.exp(predictions)
    # print(predictions,type(predictions))
    predicted_prices = predictions.tolist()
    rounded_predictions = [round(prediction) for prediction in predictions]
    # print(type(rounded_predictions),rounded_predictions)
    return render_template('prediction.html', predictions=rounded_predictions[0])
if __name__ == '__main__':
    app.run(port=5500)