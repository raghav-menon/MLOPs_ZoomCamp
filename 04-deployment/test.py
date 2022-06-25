import requests

ride = {'year':2021, 'month':4}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())