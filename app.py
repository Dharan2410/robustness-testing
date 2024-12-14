from flask import Flask, request, render_template, jsonify
import torch
from torch import nn
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Generator class definition
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 784)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x.view(-1, 1, 28, 28)

# Load the pre-trained generator model
z_dim = 100
generator = Generator(z_dim=z_dim)
generator.load_state_dict(torch.load(r"generator.pth", map_location=torch.device('cpu')))
generator.eval()

# Robustness testing function
def robustness_test(classifier_model, generator, z_dim, num_samples=100, keras=False):
    correct = 0
    total = num_samples
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, z_dim)
            adversarial_example = generator(z).view(-1, 1, 28, 28)

            # If using Keras model, convert the adversarial example to numpy and make a prediction
            if keras:
                adversarial_example_np = adversarial_example.squeeze().numpy()
                adversarial_example_np = adversarial_example_np.reshape((1, 28, 28, 1))
                output = classifier_model.predict(adversarial_example_np)
                pred = output.argmax()
            else:
                output = classifier_model(adversarial_example)
                pred = output.argmax(dim=1, keepdim=True).item()

            if pred == 1:  # Adjust this as necessary
                correct += 1
    return correct / total * 100

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return jsonify({"error": "No model file uploaded"}), 400
    model_file = request.files['model']

    # Check file extension
    if not model_file.filename.endswith('.h5'):
        return jsonify({"error": "Invalid file format. Please upload a .h5 file."}), 400

    try:
        # Load the Keras model
        classifier_model = load_model(model_file)
    except Exception as e:
        return jsonify({"error": f"Failed to load the model: {str(e)}"}), 400

    # Run the robustness test
    robustness_score = robustness_test(classifier_model, generator, z_dim)
    return jsonify({"robustness_score": robustness_score})

if __name__ == "__main__":
    app.run(debug=True)
