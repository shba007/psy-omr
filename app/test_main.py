import base64
import yaml

import pytest
from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_data = base64.b64encode(image_data).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_data}"
    
def yaml_to_json(file_path):
    # Read the YAML file
    with open(file_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    return {"name": yaml_data["name"], "choices":[{'index': i + 1, 'value': value} for i, value in enumerate(yaml_data["choices"])]}
  

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

@pytest.mark.parametrize("input_file", [
    ("EPQ-R_Test-1"),
    ("EPQ_Test-1"),
])
def test_scan(input_file:str):
    base64_image = image_to_base64(f"./tests/data/{input_file}.jpg")
    data = [base64_image]
    response = client.post("/scan", json=data)
    
    if response.status_code != 200:
        raise AssertionError(f"Request failed with {response.status_code} status code: {response.json()['detail']}")
    else:
        assert response.json()["data"] == yaml_to_json(f"./tests/data/{input_file}.yml")

  