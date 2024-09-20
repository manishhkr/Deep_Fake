from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model_name = r"C:\\FAKE_TEXT\\fake-text-detector-model"  # Use raw string notation
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to detect fake text
def detect_fake_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()  # 0 for real, 1 for fake
    return predicted_class

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    text_to_check = data.get('text', '')

    if not text_to_check:
        return jsonify({'error': 'No text provided'}), 400

    result = detect_fake_text(text_to_check)

    # Interpret the result and provide a message
    if result == 0:
        message = "Text is real"
    else:
        message = "Text is fake"

    # Response with only the message
    return jsonify({'message': message})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
