from transformers import Trainer, TrainingArguments, DistilBertForSequenceClassification, DistilBertTokenizerFast
from datasets import load_dataset, ClassLabel

# Define labels for emotions
emotions = ["sarcasm", "satisfaction", "confusion", "anger"]

# Load dataset
data = load_dataset("json", data_files="model\datasets\emotions.json")

# Map labels to integers
class_label = ClassLabel(num_classes=len(emotions), names=emotions)
def encode_labels(example):
    example['label'] = class_label.str2int(example['label'])
    return example

data = data.map(encode_labels)

# Preprocess data
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
def tokenize_data(example):
    return tokenizer(example['text'], truncation=True, padding="max_length")

tokenized_data = data.map(tokenize_data, batched=True)

# Split dataset
train_dataset = tokenized_data["train"].train_test_split(test_size=0.2)["train"]
test_dataset = tokenized_data["train"].train_test_split(test_size=0.2)["test"]

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(emotions))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./emotion-model")
tokenizer.save_pretrained("./emotion-model")
