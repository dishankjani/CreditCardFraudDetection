from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from mangum import Mangum
from flask_cors import CORS
import os
import boto3
import requests

app = Flask(__name__)
#for lambda
CORS(app)
handler = Mangum(app)
#we need to install scikit-learn to run this line
#model = pickle.load(open('model.pkl', 'rb'))

# Download the model file from S3
session = boto3.Session(
        aws_access_key_id='ASIARVBCLL7EU25LOXNW',
        aws_secret_access_key='Y6calIeS8TCP0+JkEuDL3UT2FodrkaJGOsEgkJww',
        aws_session_token="FwoGZXIvYXdzECUaDK/QdOTtN1yblMA3yiLAAeycdbftaZCjzFLAWxkDt5LpuB0FFRsKTl4OQJWxYRiA4dsqfc5jnejGGaV1rIRjzATvbw7O11DNHost7KdCsCDas4BRN/IktcTLW5AY8i0E54bOp9UTC2gQVjc70B2DjGMsaTJczUb68wVKeSJ+yproi0Qt/9TeP0G+93QGgvjAlWXbZQczn2mDSCR8PUXxdaXi7lwmvPDW7UBeUzhzmKxKnI3YMfUjpsruYE/gpsbzN++EEPWOtZOeGNnYWBi6PSjBz+KlBjIts1mzczcF/cxtgydyHPxUivh52+6oVTF6MpMRT8yO3My0IynDXpZQjz52u183",
        region_name='us-east-1'  # Replace with your desired region
    )
# Configure AWS credentials
s3_client = session.client('s3')

# Download the model file from S3
model_file = 'model.pkl'  # Replace with the name of your model file
s3_client.download_file('creditcardanomolymodeloutput', 'model.pkl', model_file)
model = pickle.load(open(model_file, 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_fraud():
    cc_num = request.form.get('cc_num')
    amt = request.form.get('amt')
    first = request.form.get('first')
    last = request.form.get('last')
    city = request.form.get('city')
    gender = request.form.get('gender')
    city_pop = request.form.get('city_pop')
    job = request.form.get('job')

    dob = request.form.get('dob')
    #dob = datetime.strptime(dob, "%Y-%m-%d").date()
    # Create the request payload as a dictionary

    x_test = np.array([[cc_num, amt, first, last, gender, city, city_pop, job, dob]])

    #Encoding categorical inputs
    cols = [2, 3, 4, 5, 7, 8]  # Column indices of categorical variables in x_test
    encoder = OrdinalEncoder()
    x_test[:, cols] = encoder.fit_transform(x_test[:, cols])

    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(x_test)

    # prediction
    result = model.predict(x_test)

    payload = {
        'cc_num': cc_num,
        'amt': amt,
        'first': first,
        'last': last,
        'city': city,
        'gender': gender,
        'city_pop': city_pop,
        'job': job,
        'dob': dob,
        'result': int(result[0])
    }

    # Make the API request
    # Extract the result from the response
    api_url = 'https://qiqdlzktog.execute-api.us-east-1.amazonaws.com/predict'
    response = requests.post(api_url, json=payload)
    response_data = response.json()  # Convert the response content to JSON

    result = response_data['result'] 
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
