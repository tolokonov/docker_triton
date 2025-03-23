from flask import Flask, request, jsonify, Response
import numpy as np
import requests
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Пример пользовательских метрик
REQUEST_COUNT = Counter('flask_request_count', 'Количество запросов в /predict_triton', ['method', 'endpoint', 'http_status'])


def normalize_data(data):
    mean = np.array([0.1307])
    std = np.array([0.3081])
    data = (data - mean) / std
    return data
    
@app.route('/predict_triton', methods=['POST'])
def predict_triton():
    try:
        data = request.json
        
        if "model_name" not in data or not data["model_name"]:
            REQUEST_COUNT.labels(method=request.method, endpoint='/predict_triton', http_status=400).inc()
            return jsonify({"error": "No model name"}), 400
        
        model_name = data["model_name"]
        # 192.168.9.10
        triton_url = f"http://192.168.9.10:8000/v2/models/{model_name}/infer"
        
        if "input" not in data or not data["input"]:
            REQUEST_COUNT.labels(method=request.method, endpoint='/predict_triton', http_status=400).inc()
            return jsonify({"error": "No input data"}), 400
        
        example_input = np.array(data['input'])
        
        batch_size = example_input.shape[0]
        example_input = example_input.reshape(batch_size, 1, 28, 28)
        normalized_data = normalize_data(example_input)
        # Проверка наличия входных данных
        if example_input is None:
            return jsonify({"error": "No input data"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    input_data = normalized_data.tolist()
    
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [batch_size, 1, 28, 28],
                "datatype": "FP32",
                "data": input_data
            }
        ]
    }
    
    try:
        response = requests.post(triton_url, json=payload)
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        REQUEST_COUNT.labels(method=request.method, endpoint='/predict_triton', http_status=500).inc()
        return jsonify({"error": f"Ошибка при запросе Triton: {str(e)}"}), 500

    output_data = np.array(result.get("outputs")[0].get("data", []))
    output_data = output_data.reshape(batch_size, 10).tolist()
    predicted_class = [int(np.argmax(x)) for x in output_data]
    REQUEST_COUNT.labels(method=request.method, endpoint='/predict_triton', http_status=200).inc()
    return jsonify({"predicted_class": predicted_class, "raw_result": result})

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)