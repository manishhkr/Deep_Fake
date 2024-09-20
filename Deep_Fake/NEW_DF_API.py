from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset
import numpy as np
import cv2
from torch import nn
import torch.nn.functional as F
from PIL import Image as pImage
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# Set the model directory


# The rest of your application logic...


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# # Load text detection model and tokenizer
# text_model_name = r"Deep_Fake"
# text_model = AutoModelForSequenceClassification.from_pretrained(text_model_name)
# text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
# Load text detection model and tokenizer
model_dir = "Deep_Fake/Deep_Fake"  # Path to the folder containing the model files

# Load the text detection model and tokenizer from the local directory
text_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
text_tokenizer = AutoTokenizer.from_pretrained(model_dir)


# Video prediction model settings
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

class ValidationDataset(Dataset):
    def __init__(self, video_path, sequence_length=60, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.count = sequence_length
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        frames = []
        for i, frame in enumerate(self.frame_extract(self.video_path)):
            faces = self.detect_faces(frame)
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Take the first detected face
                frame = frame[y:y + h, x:x + w, :]
            else:
                continue
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def predict(video_path, sequence_length):
    model = Model(2)
    path_to_model = "best_model_accuracy.pt"
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()

    video_dataset = ValidationDataset(video_path, sequence_length=sequence_length, transform=train_transforms)
    video_data = video_dataset[0]
    fmap, logits = model(video_data.to('cpu'))
    logits = F.softmax(logits, dim=1)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    prediction_text = "REAL" if prediction.item() == 1 else "FAKE"

    return prediction_text, round(confidence, 2)

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files['video']
    sequence_length = int(request.form.get('sequence_length', 60))

    video_path = os.path.join('uploads', video.filename)
    video.save(video_path)

    # Perform the prediction
    prediction_text, confidence = predict(video_path, sequence_length)

    # Clean up the saved video after prediction
    os.remove(video_path)

    return jsonify({"prediction": prediction_text, "confidence": confidence})

# Function to detect fake text
def detect_fake_text(text):
    inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = text_model(**inputs)
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
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=8000)
