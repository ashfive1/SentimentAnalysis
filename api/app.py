from flask import Flask, request, jsonify
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

# Load the model
model = DistilBertForSequenceClassification.from_pretrained("./emotion-model")
tokenizer = DistilBertTokenizerFast.from_pretrained("./emotion-model")
labels = ["satisfaction", "sarcasm", "confusion", "anger"]

# Initialize Flask
app = Flask(__name__)

# Endpoint to analyze emotion
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data["text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_emotion = labels[predicted_class]
    
    return jsonify({"predicted_emotion": predicted_emotion})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
