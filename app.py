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
        aws_access_key_id='ASIARVBCLL7ER4TNZRNV',
        aws_secret_access_key='qVIhyyrF/hAsNBPNHl+RS33lzh2Tr6y9Y/mog+FI',
        aws_session_token="FwoGZXIvYXdzEO///////////wEaDM0cigjZgXuKaxnVPCLAAb/g0iZJghK+dBID5R8ov/qIKrDgiyw0wERwlDtj6wgseQVCSrGjhJYKDG3b8p/F2gS/OYEq5ZXpg4uq5lXmcgGHU3D2qU9KPSupkhlym9XFJWtgvJW8v3edQarKdl6roD5JIOKe9TowDbJA0PLcn+O9sgItTpqnMYrdLqBS4CTEHv6eQH1+876qxv8GvqBEJU9gIOPGn9lQ1pif22ck7NPzfMgACGY/y12EPrEKILTULR5xcpK8YnTA8wHFk9+5YCi5iY+mBjItLrIWT1UY39yYKWn7O5YbWN4wFQNPPiwWvZDp4Iy011nXQX3qtz7t19VPc+Rr",
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
