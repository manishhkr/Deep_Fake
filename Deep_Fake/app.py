# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch

# # Load the model and tokenizer
# model_name = "C:\\FAKE_TEXT\\fake-text-detector-model\\pytorch_model.bin"  # Replace with the path to your model on Hugging Face Hub or local path
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Function to detect fake text
# def detect_fake_text(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     predicted_class = logits.argmax(dim=-1).item()
#     return predicted_class

# # Example usage
# text_to_check = "Scientists have proven that the moon landing was staged by Hollywood in the 1960s."
# result = detect_fake_text(text_to_check)

# # Interpret the result
# if result == 1:  # Assuming 1 is fake and 0 is real
#     print("The text is likely fake.")
# else:
#     print("The text is likely real.")


from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import sentencepiece as spm


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
    predicted_class = logits.argmax(dim=-1).item()
    return predicted_class

# Example usage
#text_to_check = "Scientists have proven that the moon landing was staged by Hollywood in the 1960s."
text_to_check="Generally peoples have 15 fingers in hand"
result = detect_fake_text(text_to_check)

# Interpret the result
if result == 1:  # Assuming 1 is fake and 0 is real
    print("The text is likely fake.")
else:
    print("The text is likely real.")

