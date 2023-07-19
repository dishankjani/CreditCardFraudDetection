import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style


import boto3
import pickle

# Your AWS credentials
aws_access_key_id = 'ASIARVBCLL7ERPRX2EMK'
aws_secret_access_key = 'gEzas+hPLcQXAC0phkGBwZj7MG9Udofcnx84OY/d'
aws_session_token = 'FwoGZXIvYXdzEBYaDNC6Y5zH8CiM8gombCLAAXIlwC2Pi78kWkRY8EN9aPjHzNqHWgsm0uDoQju6D5LGDJqZzN3fNyOBNvS3n+1OirCMH6mRSzthbTOL+FQvBKZmDQ8wbDelRSwUrclq1rzjd0CA6Lt442nbg2Xv55jj2FZFn+Rc8Q9kaXNE0YeOYBIVjTBHjd8cAH0KdBdft0Ey/reeFSQ+0+wRmlapxiOXF4raGjopCvhozbjRDh34AYbwRyT6tscll3fbkDDtqvJ9iv25r4o8R+s/0DCWhPT2wCjUtN+lBjItemld2n7XUeDjk5SvpgTmeRytFzwteYpEwkxpNIaiEL5/1nfh7snrKdVxKFo4'  # If using temporary credentials (Optional)

# Create a Boto3 session with the provided credentials
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    region_name='us-east-1'  # Replace with your desired region
)

# Configure AWS credentials
s3_client = session.client('s3')

#response = s3_client.get_object(Bucket='datasetfortraining', Key='fraudTest.csv')


testData = pd.read_csv('https://datasetfortraining.s3.amazonaws.com/fraudTest.csv')

#response1 = s3_client.get_object(Bucket='datasetfortraining', Key='fraudTrain.csv')
trainData = pd.read_csv("https://datasetfortraining.s3.amazonaws.com/fraudTrain.csv")



data = pd.concat([trainData, testData], axis = 0)


# Resetting the index
data.reset_index(inplace = True)

data = data.drop(['index', 'Unnamed: 0'], axis = 1)

# doing some prelimenary adjustments to data set
data = data.drop('trans_num', axis=1)
data = data.drop('street', axis=1)
data = data.drop('zip', axis=1)
data = data.drop('lat', axis=1)
data = data.drop('long', axis=1)
data = data.drop('unix_time', axis=1)
data = data.drop('merch_lat', axis=1)
data = data.drop('merch_long', axis=1)

data = data.drop('trans_date_trans_time', axis=1)
data = data.drop('merchant', axis=1)
data = data.drop('category', axis=1)


data = data.drop('state', axis=1)



y = data['is_fraud']
x = data.drop('is_fraud', axis=1)

# Encoding the categorical columns
from sklearn.preprocessing import OrdinalEncoder
cols = ['first', 'last', 'gender', 'city', 'job', 'dob']
encoder = OrdinalEncoder()
x[cols] = encoder.fit_transform(x[cols])


# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)


y = data[['is_fraud']].values

print("Performed preprocessing..")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

model = LogisticRegression()
model.fit(x_train, y_train)
print("Performed training...")
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

pickle.dump(model,open('model.pkl','wb'))



file_path = 'model.pkl'
key_name = 'model.pkl'
bucket_name = 'creditcardanomolymodeloutput'
s3_client.upload_file(file_path, bucket_name, key_name)
print("everything done")