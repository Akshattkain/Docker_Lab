from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import pickle

app = Flask(__name__, static_folder='statics')

class BanknoteModel(nn.Module):
    def __init__(self):
        super(BanknoteModel, self).__init__()
        self.layer1 = nn.Linear(4, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# Load model and scaler
model = BanknoteModel()
model.load_state_dict(torch.load('banknote_model.pth'))
model.eval()

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        variance = float(data['variance'])
        skewness = float(data['skewness'])
        kurtosis = float(data['kurtosis'])
        entropy = float(data['entropy'])
        
        # Preprocess
        features = np.array([[variance, skewness, kurtosis, entropy]])
        features_scaled = scaler.transform(features)
        input_tensor = torch.FloatTensor(features_scaled)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probability = output.item()
            prediction = "AUTHENTIC" if probability > 0.5 else "FAKE"
            confidence = probability if probability > 0.5 else 1 - probability
        
        return jsonify({
            "prediction": prediction,
            "confidence": f"{confidence:.1%}"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)