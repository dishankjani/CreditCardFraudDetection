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

# Load the API Gateway URI from the environment variable
api_url = os.environ.get('ApiGatewayEndpoint')
# Download the model file from S3
session = boto3.Session(
        aws_access_key_id='ASIARVBCLL7E3UTOJKPF',
        aws_secret_access_key='g7qYhj2KlyFTAUb7GIhDQdBG6YIIT4wqkBAs72qs',
        aws_session_token="FwoGZXIvYXdzEDIaDJwzd42KqPmCridAsCLAATSz4SVT75quaJPcuzFrrnFccInOOS4xch+vgFV+m8w0LS8ELjyu30mXbyrY9uTaj2UOd7tTC1Nnu8InIgkYq/pHZSHlnZN8XbkFk6j2gwDuJ+LxGclP0BjnNyoVtUH9LCiyeyRssyc4fvkBEH+DN+p5xEby3SQIodaieWdIsFxppQhnFAa6WHTM91Qx2ny56RSImZLJoP4qrYjDMIujcaR4pkg7HED3UV8xEqqRcxgw5e86KgNZLrMML3mFBl4Coyjwx+WlBjIt4kI8tXX5MuSXcDrYFVmW0bEqKBtWjpc8X6BYpNnR3obTYpT6CvTCH53yMu/3",
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
    #api_url = 'https://qiqdlzktog.execute-api.us-east-1.amazonaws.com/predict'
    response = requests.post(api_url, json=payload)
    response_data = response.json()  # Convert the response content to JSON

    result = response_data['result'] 
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
