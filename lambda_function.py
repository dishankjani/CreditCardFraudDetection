import json
import numpy as np
import boto3
from urllib.parse import parse_qs
import os

def min_max_scale(data):
    data = data.astype(float)  # Convert data to float type
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    scaled_data = (data - min_vals) / (max_vals - min_vals)
    
    return scaled_data, min_vals, max_vals
    
def ordinal_encode(data):
    categories = []
    encoded_data = np.zeros_like(data, dtype=int)
    
    for i in range(data.shape[1]):
        unique_values = np.unique(data[:, i])
        categories.append(unique_values)
        encoded_data[:, i] = np.searchsorted(unique_values, data[:, i])
    
    return encoded_data, categories

def process_data(x_test):
    # Assuming columns to encode are: 'first', 'last', 'gender', 'city', 'job', 'dob'
    cols_to_encode = [2, 3, 4, 5, 7, 8]
    
    # Perform ordinal encoding for selected columns
    encoded_data, categories = ordinal_encode(x_test[:, cols_to_encode])
    
    # Replace the original columns with the encoded values
    x_test_encoded = np.concatenate((x_test[:, :2], encoded_data, x_test[:, 10:]), axis=1)
    
    return x_test_encoded, categories

def send_sns_message(message):
    topic_arn = "arn:aws:sns:" + os.environ['AWS_REGION'] + ":" + os.environ['AWS_ACCOUNT_ID'] + ":MySnsTopicArn"  # Replace with your SNS topic ARN
    sns_client = boto3.client('sns')
    sns_client.publish(
        TopicArn=topic_arn,
        Message=message
    )
    
def send_sqs_message(queue_url, message_body):
    sqs_client = boto3.client('sqs')
    response = sqs_client.send_message(
        QueueUrl=queue_url,
        MessageBody=message_body
    )
    return response

def lambda_handler(event, context):
    S3_BUCKET_NAME = 'cloudtermassignment'
    
    # Your AWS credentials
    session = boto3.Session(
    aws_access_key_id='ASIARVBCLL7ER4TNZRNV',
    aws_secret_access_key='qVIhyyrF/hAsNBPNHl+RS33lzh2Tr6y9Y/mog+FI',
    aws_session_token="FwoGZXIvYXdzEO///////////wEaDM0cigjZgXuKaxnVPCLAAb/g0iZJghK+dBID5R8ov/qIKrDgiyw0wERwlDtj6wgseQVCSrGjhJYKDG3b8p/F2gS/OYEq5ZXpg4uq5lXmcgGHU3D2qU9KPSupkhlym9XFJWtgvJW8v3edQarKdl6roD5JIOKe9TowDbJA0PLcn+O9sgItTpqnMYrdLqBS4CTEHv6eQH1+876qxv8GvqBEJU9gIOPGn9lQ1pif22ck7NPzfMgACGY/y12EPrEKILTULR5xcpK8YnTA8wHFk9+5YCi5iY+mBjItLrIWT1UY39yYKWn7O5YbWN4wFQNPPiwWvZDp4Iy011nXQX3qtz7t19VPc+Rr",
    region_name='us-east-1'  # Replace with your desired region
    )
    
    # Configure AWS credentials
    s3_client = session.client('s3')
    request_body = event['body']
    print(request_body)
    
    # Assuming the body contains JSON data, you can parse it
    data = json.loads(request_body)
    cc_num = data['cc_num']
    amt = data['amt']
    first = data['first']
    last = data['last']
    city = data['city']
    gender = data['gender']
    city_pop = data['city_pop']
    job = data['job']
    dob = data['dob']
    result = data.get('result')  # Use data.get('result') to handle missing 'result' key
    
    # Create the original data dictionary
    x_test1 = {
        'cc_num': cc_num,
        'amt': amt,
        'first': first,
        'last': last,
        'gender': gender,
        'city': city,
        'city_pop': city_pop,
        'job': job,
        'dob': dob,
        'result': result
    }
    
    # Create the original data array
    x_test = np.array([[cc_num, amt, first, last, gender, city, city_pop, job, dob]])

    x_test_encoded, categories = process_data(x_test)
    
    # Apply scaling to the encoded data
    x_test_scaled, min_vals, max_vals = min_max_scale(x_test_encoded)
    
    # Convert data dictionary to JSON string
    x_test_json = json.dumps(x_test1)
    
    # Save the data to S3 bucket
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key='data.json',
        Body=x_test_json
    )
    
    print(result)
    
    fraud_result = "Not defined"
    
    if result == 1:
        fraud_result = "Fraud"
    elif result == 0:
        fraud_result = "Not Fraud"
        message = f"Fraudulent transaction detected for credit card. Please contact your bank immediately."
        send_sns_message(message)
        #sqs_queue_url = 'https://sqs.us-east-1.amazonaws.com/113888681929/creditcardfraud'  # Replace with your SQS queue URL
        #message = f"Fraudulent transaction detected for credit card {cc_num}. Please contact your bank immediately."
        #send_sqs_message(sqs_queue_url, message)
    print(fraud_result)
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Data received successfully',
            'result': fraud_result
        })
    }