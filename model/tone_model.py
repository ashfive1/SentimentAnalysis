from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

# Load fine-tuned model and tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("./emotion-model")
model = DistilBertForSequenceClassification.from_pretrained("./emotion-model")

# Define emotion labels
emotion_labels = ["sarcasm", "satisfaction", "confusion", "anger"]

def analyze_tone(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return emotion_labels[predicted_class]
