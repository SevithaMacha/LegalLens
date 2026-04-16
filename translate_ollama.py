import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": "Translate to Telugu: Hello, how are you?",
        "stream": False
    }
)

result = response.json()
print("Result:", result["response"])
