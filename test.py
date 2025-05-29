import requests

data = {"input": "i am happy to help you"}
response = requests.post("http://127.0.0.1:5000/translate", json=data)
print("Server Response:", response.json())
