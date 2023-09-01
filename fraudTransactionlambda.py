import json

def lambda_handler(event, context):
    message = f"Error Reported to Bank of the Given Credit Card"
    print(message)
    response = {
        'statusCode': 200,
        'body': json.dumps({'message': message})
    }

    return response