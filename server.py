from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    input_text = data.get("input", "")
    result = translator(input_text)
    translated = result[0]['translation_text']
    return jsonify({"translation": translated})

@app.route('/')
def home():
    return "ASL-to-English Cloud Server is running!"

if __name__ == "__main__":
    print("Starting Server")
    app.run(debug=True)
