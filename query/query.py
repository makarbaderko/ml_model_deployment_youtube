import requests

resp = requests.post(
    "http://127.0.0.1:80/predict",
    json={"text":input()}
)

print(resp.json())
