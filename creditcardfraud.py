import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style


import boto3
import pickle

session = boto3.Session(region_name='us-east-1')
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